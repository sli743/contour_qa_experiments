import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from .models import VAE, CVAE, VQVAE, HybridVAEUNet, GMVAE2_3D, VAE_CBAM
from .losses import calc_recon_loss, calc_kld, calc_mmd, EmbeddingLoss, gmvae_kld_loss
from utils.lr_scheduler import LR_scheduler
from utils.helper import save_model, StopCriteria_LossImprove
import tqdm
import wandb

def compute_slice_from_img(img, cf, mode='label', axis = 'axial'):
    if len(img.shape) == 5:
        img = img[0]
    if cf['train_label']:
        if mode == 'prob':
            img = torch.argmax(img, dim=0).to(torch.float32)
        elif mode == 'label':
            if len(img.shape) == 4:
                assert img.shape[0] == 1
                img = img[0]
        if len(img.shape) == 3:
            if axis == 'axial':
                img = img[img.shape[0] // 2]
            elif axis == 'coronal':
                img = img[:, img.shape[1] // 2, :]
            elif axis == 'sagittal':
                img = img[:, :, img.shape[2] // 2]
    else:
        img = img[0]
        if len(img.shape) == 3:
            if axis == 'axial':
                img = img[img.shape[0] // 2]
            elif axis == 'coronal':
                img = img[:, img.shape[1] // 2, :]
            elif axis == 'sagittal':
                img = img[:, :, img.shape[2] // 2]
    print(f"Sample slice shape: {img.shape}, max: {img.max().item()}, min: {img.min().item()}")
    return img

def crop_to_size(tensor, target_size):
    """Crop tensor to target spatial size, assuming [batch, channels, *spatial]."""
    current_size = tensor.shape[2:]
    if current_size != target_size:
        start = [(c - t) // 2 for c, t in zip(current_size, target_size)]
        end = [s + t for s, t in zip(start, target_size)]
        tensor = tensor[:, :, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    return tensor

def NormalizeImage(x, cf):
    if cf['train_label']:
        # case 1: input are images
        num_of_labels = cf['out_ch']
        x = x
    else:
        # case 2: input are images, pixel values 0-255
        x = x / 256
    return x

def UnNormalizeImage(x, cf):
    if cf['train_label']:
        x = x
    else:
        x = x * 256
        x = np.round(x)
        x = x.astype(np.uint8)
    return x

def apply_label_mask(x, cf):
    label_mask = cf.get('label_mask', None)
    # For rois not included in the label mask, just combine them into background class
    if label_mask is not None:
        x1 = x.clone()
        labels = torch.unique(x1)
        label_dict = {label_mask[i]: i for i in range(len(label_mask))}
        for l in labels:
            if l.item() not in label_mask:
                x1[x1 == l.item()] = 0
            else:
                x1[x1 == l.item()] = label_dict[l.item()]
        x = x1
    return x

import torch

def to_label_map(vol, cf, is_prediction=False, threshold=0.5):
    """
    Convert x or x_hat to canonical integer label map [B,1,D,H,W].

    For your left-lung binary setup:
      background = 0
      left lung = 1

    Parameters
    ----------
    vol : torch.Tensor
        x or x_hat
    cf : dict
    is_prediction : bool
        True for x_hat, False for x
    threshold : float
        Used only for binary prediction logits/probabilities

    Returns
    -------
    torch.Tensor
        Integer label map [B,1,D,H,W]
    """
    if not cf.get("train_label", False):
        raise ValueError("to_label_map is intended for label training / label recon only.")

    # Ground truth x
    if not is_prediction:
        # x is usually already [B,1,D,H,W]
        if vol.ndim != 5:
            raise ValueError(f"Expected GT shape [B,1,D,H,W], got {vol.shape}")
        # binary foreground/background
        return (vol > 0).long()

    # Prediction x_hat
    if vol.ndim != 5:
        raise ValueError(f"Expected prediction shape [B,C,D,H,W], got {vol.shape}")

    c = vol.shape[1]

    # Binary model: out_ch = 1, logits expected
    if c == 1:
        prob = torch.sigmoid(vol)
        return (prob > threshold).long()

    # Multiclass model: out_ch > 1
    return torch.argmax(vol, dim=1, keepdim=True).long()

def log_roi_slices_to_wandb(
    x,
    x_hat,
    cf,
    epoch,
    dataset,
    roi_table,
    label_names,
    axis="axial",
    max_rois=None,
    img_name=None,
    threshold=0.5,
):
    """
    x:     GT, usually [B,1,D,H,W]
    x_hat: prediction, [B,C,D,H,W] or already [B,1,D,H,W]
    """
    with torch.no_grad():
        x_lbl = to_label_map(x, cf, is_prediction=False, threshold=threshold)
        xhat_lbl = to_label_map(x_hat, cf, is_prediction=True, threshold=threshold)

        x0 = x_lbl[:1]
        xhat0 = xhat_lbl[:1]

        roi_ids = sorted(label_names.keys())
        if max_rois is not None:
            roi_ids = roi_ids[:max_rois]

        for k in roi_ids:
            x_roi = (x0 == k).float()
            xhat_roi = (xhat0 == k).float()

            x_slice = compute_slice_from_img(x_roi, cf, mode="label", axis=axis)
            xhat_slice = compute_slice_from_img(xhat_roi, cf, mode="label", axis=axis)
            diff_slice = torch.abs(x_slice - xhat_slice)

            roi_table.add_data(
                epoch,
                int(k),
                label_names[k],
                dataset,
                axis,
                wandb.Image(x_slice.cpu().numpy(), caption=f"GT | {label_names[k]}"),
                wandb.Image(xhat_slice.cpu().numpy(), caption=f"Pred | {label_names[k]}"),
                wandb.Image(diff_slice.cpu().numpy(), caption=f"Diff | {label_names[k]}"),
                img_name,
            )

def make_slice_triplet(x, x_hat, cf, axis="axial", threshold=0.5):
    """
    Returns x_slice, xhat_slice, diff_slice for binary or multiclass label data.
    All outputs are 2D tensors.
    """
    with torch.no_grad():
        x_lbl = to_label_map(x, cf, is_prediction=False, threshold=threshold).float()
        xhat_lbl = to_label_map(x_hat, cf, is_prediction=True, threshold=threshold).float()

        x_slice = compute_slice_from_img(x_lbl[:1], cf, mode="label", axis=axis)
        xhat_slice = compute_slice_from_img(xhat_lbl[:1], cf, mode="label", axis=axis)
        diff_slice = torch.abs(x_slice - xhat_slice)

    return x_slice, xhat_slice, diff_slice

def train_model(train_loader, val_loader, cf):
    wandb.init(project="VAE-Variant-Cardiac", entity="coral")
    wandb.run.name = cf['exp_name']

    DEFAULT_LABEL_NAMES = {
        0: "Background",
        1: "Aortic_Valve",
        2: "LAD",
        3: "Tricuspid_Valve",
        4: "Mitral_Valve",
        5: "Pulmonic_Valve",
        6: "Right_Atrium",
        7: "Right_Ventricle",
        8: "Left_Atrium",
        9: "Left_Ventricle",
    }

    if 'label_names' in cf and cf['label_names'] is not None:
        if cf.get('label_mask', None) is None:
            # YAML keys may be strings → cast to int
            label_names = {int(k): v for k, v in cf['label_names'].items()}
        else:
            label_names = {int(i): cf['label_names'][cf['label_mask'][i]] for i in range(len(cf['label_mask']))}
    else:
        label_names = DEFAULT_LABEL_NAMES.copy()

    if cf['train_label'] and cf['recon_loss'] not in ['dice', 'dice_entropy', 'gaussian_nll', 'dice_focal', 'hausdorff', 'dice_bce']:
        raise ValueError(f"Label training requires recon_loss in ['dice', 'dice_entropy', 'gaussian_nll', 'dice_focal', 'hausdorff', 'dice_bce'], got {cf['recon_loss']}")
    if cf['model'] == 'paired_vae_label' and cf['recon_loss'] != 'gaussian_nll':
        raise ValueError("Paired VAE requires gaussian_nll_loss")

    n_train_images = len(train_loader.dataset)
    n_val_images = len(val_loader.dataset) if val_loader is not None else 0
    weight_mode = cf.get('weight_mode', 'uniform')

    print(f"\n > Model type: {cf['model']}, dim: {cf['dim']}, train_label: {cf['train_label']}")
    print(f"  Training images: {n_train_images}, Validation images: {n_val_images}")
    device = cf['device']

    embedding = (cf['model'] == 'reverse_vae')
    if cf['model'] in ['beta_vae', 'vae', 'infovae']:
        model = VAE(cf, embedding=embedding).to(device=device)
    elif cf['model'] == 'cvae':
        model = CVAE(cf, embedding=embedding).to(device)
    elif cf['model'] == 'vqvae':
        model = VQVAE(cf, embedding=embedding).to(device)
    elif cf['model'] == 'hybrid_vae_unet':
        model = HybridVAEUNet(cf, embedding=embedding).to(device)
    elif cf['model'] == 'GMVAE2_3D':
        model = GMVAE2_3D(cf, embedding=embedding).to(device)
    elif cf['model'] == 'vae_cbam':
        model = VAE_CBAM(cf, embedding=embedding).to(device)
    else:
        model = VAE(cf, embedding=embedding).to(device=device)

    ckpt_dir = os.path.join(cf['output_folder'], 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params_trainable}")

    if cf['resume']:
        model.load_state_dict(torch.load(cf['resume_path'], map_location=device))
        print(f"\n > Resume training with: {cf['resume_path']}")

    if cf['model'] in ['intro_vae', 'soft_intro_vae', 'reverse_vae']:
        siv_cf = cf['soft_intro_vae']
        optimizer_e = torch.optim.Adam(model.encoder.parameters(), lr=siv_cf['lr_e'], weight_decay=cf['weight_decay'])
        optimizer_d = torch.optim.Adam(model.decoder.parameters(), lr=siv_cf['lr_d'], weight_decay=cf['weight_decay'])
        scheduler_e = LR_scheduler(optimizer_e, scheduler_type=cf['lr_scheduler']['type'], 
                                    gamma=cf['lr_scheduler']['factor'], tol_epochs=cf['num_epochs'],
                                    spe_arg=cf['lr_scheduler']['spe_arg'])
        scheduler_d = LR_scheduler(optimizer_d, scheduler_type=cf['lr_scheduler']['type'],
                                     gamma=cf['lr_scheduler']['factor'], tol_epochs=cf['num_epochs'],
                                     spe_arg=cf['lr_scheduler']['spe_arg'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cf['init_lr'], weight_decay=cf['weight_decay'])
        scheduler = LR_scheduler(optimizer, scheduler_type=cf['lr_scheduler']['type'], 
                                 gamma=cf['lr_scheduler']['factor'], tol_epochs=cf['num_epochs'],
                                 spe_arg=cf['lr_scheduler']['spe_arg'])

    if cf['early_stopping']['type'] == 'loss_improve':
        stop_criterion = StopCriteria_LossImprove(
            query_len=cf['early_stopping']['patience'],
            num_min_epoch=cf['early_stopping']['min_epoch'],
            min_improve=cf['early_stopping']['min_improve']
        )
    else:
        stop_criterion = None

    tb_writer = SummaryWriter(log_dir=os.path.join(cf['output_folder'], 'tensorboard_' + cf['timestamp']))

    if cf['model'] in ['vae', 'beta_vae', 'paired_vae_label', 'cvae', 'infovae', 'hybrid_vae_unet', 'GMVAE2_3D', 'vae_cbam']:
        cyc_annealing_batchs = cf['annealing_cyc_epochs'] * len(train_loader)
        increasing_batchs = int(cf['annealing_cyc_epochs'] * cf['annealing_inc_ratio']) * len(train_loader)

    if cf['model'] == 'reverse_vae':
        embedding_loss = EmbeddingLoss()

    best_loss = float("inf")
    global_step = 0
    my_table = wandb.Table(columns=["epoch", "prediction_slice_train", "slice_train", "diff_slice_train", "prediction_slice_test", "slice_test", "diff_slice_test", "slice_randrecon"])
    roi_table = wandb.Table(
        columns=[
            "epoch",
            "roi_id",
            "roi_name",
            "dataset",
            "axis",
            "x_slice",
            "xhat_slice",
            "diff_slice",
            "image_name",
        ]
    )

    target_size = cf['ori_size']
    label_mask = cf.get('label_mask', None)

    for epoch in range(cf['num_epochs']):
        model.train()
        train_recon_losses = []
        train_kld_losses = []
        train_dice_scores = [[] for _ in range(cf['out_ch'])]
        train_dice_focal_losses = [[] for _ in range(cf['out_ch'])]
        train_image_count = 0

        if cf['model'] == 'paired_vae_label':
            train_recon_loss_classes = [[] for _ in range(cf['out_ch'] // 2)]
        else:
            train_recon_loss_classes = None

        pbar = tqdm.tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{cf['num_epochs']}", unit='b')

        for input_proj in pbar:
            x = input_proj['proj_image'].to(device=device)
            xname = input_proj['proj_name']
            # print(f"xname is {xname}")
            x = NormalizeImage(x, cf)
            x999 = x
            x = apply_label_mask(x, cf)
            # check if x999 == 11 is same as x == 2
            # print(f"nnz of x999 in LungR is {(x999==11).sum().item()}")
            # print(f"nnz of x in LungR is {(x==2).sum().item()}")
            # if not torch.equal(x999 == 11, x == 2):
            #     raise ValueError(f"Label mask application incorrect. ")
            # if x.shape[1] != cf['in_ch'] or tuple(x.shape[2:]) != tuple(cf['ori_size']):
            #     raise ValueError(f"Input shape {x.shape} does not match expected [batch, {cf['in_ch']}, {cf['ori_size']}]")

            b_size = x.shape[0]
            if cf['model'] == 'ae':
                # --------------------------------------------------
                # Post-DAE compatible input/target handling
                # --------------------------------------------------
                x_in = input_proj['proj_image'].to(device=device)
                x_target = input_proj.get('target_image', input_proj['proj_image']).to(device=device)

                x_in = NormalizeImage(x_in, cf)

                # If Post-DAE is active, x_in is already binary {0,1}
                if cf['pre_processing'].get('post_dae_train', False) and cf.get('train', False):
                    x_in = (x_in > 0).float()
                else:
                    x_in = apply_label_mask(x_in, cf)

                x_target = NormalizeImage(x_target, cf)
                x_target = apply_label_mask(x_target, cf)
                print("x_in unique:", torch.unique(x_in))
                print("x_target unique:", torch.unique(x_target))

                b_size = x_in.shape[0]

                x_hat, _, _, _ = model(x_in, deterministic=True)
                x_hat = crop_to_size(x_hat, target_size)

                recon_loss = calc_recon_loss(
                    x_hat,
                    x_target,
                    loss_type=cf['recon_loss'],
                    reduction='mean',
                    weight_mode=weight_mode,
                    return_per_class=True,
                    alpha=cf.get('alpha', 0.5),
                    epoch=epoch + 1
                )

                if cf['train_label'] and cf['recon_loss'] == 'dice_bce':
                    target = (x_target > 0).float()
                    prob = torch.sigmoid(x_hat)
                    intersection = (prob * target).sum(dim=(2, 3, 4))
                    denominator = prob.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator  # [B,1]

                    for i in range(b_size):
                        train_dice_focal_losses[0].append(recon_loss[i, 0].item())
                        train_dice_scores[0].append(dice_score[i, 0].item())
                        if dice_score[i, 0] < 0 or dice_score[i, 0] > 1:
                            print(
                                f"Invalid Dice score for class {label_names[1]}: "
                                f"{dice_score[i, 0]}, intersection: {intersection[i, 0]}, denominator: {denominator[i, 0]}"
                            )

                elif cf['train_label'] and cf['recon_loss'] in ['dice', 'dice_entropy', 'dice_focal']:
                    x_one_hot = F.one_hot(
                        x_target.squeeze(1).long(),
                        num_classes=x_hat.shape[1]
                    ).permute(0, 4, 1, 2, 3).float()

                    intersection = (x_hat * x_one_hot).sum(dim=(2, 3, 4))
                    denominator = (x_hat.sum(dim=(2, 3, 4)) + x_one_hot.sum(dim=(2, 3, 4))) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator

                    for i in range(b_size):
                        for k in range(cf['out_ch']):
                            train_dice_focal_losses[k].append(recon_loss[i, k].item())
                            train_dice_scores[k].append(dice_score[i, k].item())
                            if dice_score[i, k] < 0 or dice_score[i, k] > 1:
                                print(
                                    f"Invalid Dice score for class {label_names[k]}: "
                                    f"{dice_score[i, k]}, intersection: {intersection[i, k]}, denominator: {denominator[i, k]}"
                                )

                recon_loss_per_image = calc_recon_loss(
                    x_hat,
                    x_target,
                    loss_type=cf['recon_loss'],
                    reduction='none',
                    weight_mode=weight_mode,
                    return_per_class=False,
                    alpha=cf.get('alpha', 0.5),
                    epoch=epoch + 1
                )

                train_recon_losses.extend(recon_loss_per_image.tolist())
                train_kld_losses.extend([0.0] * b_size)

                total_loss = calc_recon_loss(
                    x_hat,
                    x_target,
                    loss_type=cf['recon_loss'],
                    reduction='batch_mean',
                    weight_mode=weight_mode,
                    return_per_class=False,
                    alpha=cf.get('alpha', 0.5),
                    epoch=epoch + 1
                )

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Keep these for logging / slice visualization later in the epoch
                x = x_target

            elif cf['model'] == 'beta_vae':
                beta = cf.get('beta', 1.0)
                if cf['beta_cyc_annealing']:
                    beta = cf['kld_weight'] * min(1, (global_step % cyc_annealing_batchs) / increasing_batchs)
                else:
                    beta = cf['kld_weight']

                x_hat, _, mu, log_var = model(x)
                x_hat = crop_to_size(x_hat, target_size)
                recon_loss = calc_recon_loss(x_hat, x, loss_type=cf['recon_loss'], reduction='mean', 
                                                weight_mode=weight_mode, return_per_class=True, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                recon_loss_per_image = calc_recon_loss(x_hat, x, loss_type=cf['recon_loss'], reduction='none', 
                                                      weight_mode=weight_mode, return_per_class=False, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                train_recon_losses.extend(recon_loss_per_image.tolist())
                
                if cf['train_label'] and cf['recon_loss'] == 'dice_bce':
                    target = (x > 0).float()
                    prob = torch.sigmoid(x_hat)
                    intersection = (prob * target).sum(dim=(2,3,4))
                    denominator = prob.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator   # shape [B,1]
                    for i in range(b_size):
                        train_dice_focal_losses[0].append(recon_loss[i, 0].item())
                        train_dice_scores[0].append(dice_score[i, 0].item())
                        if dice_score[i, 0] < 0 or dice_score[i, 0] > 1:
                            print(f"Invalid Dice score for class {label_names[1]}: {dice_score[i, 0]}, intersection: {intersection[i, 0]}, denominator: {denominator[i, 0]}")

                if cf['train_label'] and cf['recon_loss'] in ['dice', 'dice_entropy', 'dice_focal']:
                    x_one_hot = F.one_hot(x.squeeze(1).long(), num_classes=x_hat.shape[1]).permute(0, 4, 1, 2, 3).float()
                    intersection = (x_hat * x_one_hot).sum(dim=(2, 3, 4))
                    denominator = (x_hat.sum(dim=(2, 3, 4)) + x_one_hot.sum(dim=(2, 3, 4))) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator
                    for i in range(b_size):
                        for k in range(cf['out_ch']):
                            train_dice_focal_losses[k].append(recon_loss[i, k].item())
                            train_dice_scores[k].append(dice_score[i, k].item())
                            if dice_score[i, k] < 0 or dice_score[i, k] > 1:
                                print(f"Invalid Dice score for class {label_names[k]}: {dice_score[i, k]}, intersection: {intersection[i, k]}, denominator: {denominator[i, k]}")
                kld_loss = calc_kld(mu, log_var, reduction='none', beta=beta)
                train_kld_losses.extend(kld_loss.tolist())
                total_loss = recon_loss_per_image.mean() + kld_loss.mean()
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            elif cf['model'] == 'cvae':
                if cf['beta_cyc_annealing']:
                    beta = cf['kld_weight'] * min(1, (global_step % cyc_annealing_batchs) / increasing_batchs)
                else:
                    beta = cf['kld_weight']

                labels = x.squeeze(1).long().view(b_size, -1)
                cond = torch.zeros(b_size, cf['out_ch'], device=device)
                for i in range(b_size):
                    counts = torch.bincount(labels[i], minlength=cf['out_ch']).float()
                    cond[i] = counts / counts.sum()
                cond = cond.to(device)

                x_hat, _, mu, log_var = model(x, cond)
                x_hat = crop_to_size(x_hat, target_size)
                recon_loss = calc_recon_loss(x_hat, x, loss_type=cf['recon_loss'], reduction='mean', 
                                            weight_mode=weight_mode, return_per_class=True, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                recon_loss_per_image = calc_recon_loss(x_hat, x, loss_type=cf['recon_loss'], reduction='none', 
                                                      weight_mode=weight_mode, return_per_class=False, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                train_recon_losses.extend(recon_loss_per_image.tolist())
                
                if cf['train_label'] and cf['recon_loss'] == 'dice_bce':
                    target = (x > 0).float()
                    prob = torch.sigmoid(x_hat)
                    intersection = (prob * target).sum(dim=(2,3,4))
                    denominator = prob.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator   # shape [B,1]
                    for i in range(b_size):
                        train_dice_focal_losses[0].append(recon_loss[i, 0].item())
                        train_dice_scores[0].append(dice_score[i, 0].item())
                        if dice_score[i, 0] < 0 or dice_score[i, 0] > 1:
                            print(f"Invalid Dice score for class {label_names[1]}: {dice_score[i, 0]}, intersection: {intersection[i, 0]}, denominator: {denominator[i, 0]}")

                if cf['train_label'] and cf['recon_loss'] in ['dice', 'dice_entropy', 'dice_focal']:
                    x_one_hot = F.one_hot(x.squeeze(1).long(), num_classes=x_hat.shape[1]).permute(0, 4, 1, 2, 3).float()
                    intersection = (x_hat * x_one_hot).sum(dim=(2, 3, 4))
                    denominator = (x_hat.sum(dim=(2, 3, 4)) + x_one_hot.sum(dim=(2, 3, 4))) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator
                    for i in range(b_size):
                        for k in range(cf['out_ch']):
                            train_dice_focal_losses[k].append(recon_loss[i, k].item())
                            train_dice_scores[k].append(dice_score[i, k].item())
                            if dice_score[i, k] < 0 or dice_score[i, k] > 1:
                                print(f"Invalid Dice score for class {label_names[k]}: {dice_score[i, k]}, intersection: {intersection[i, k]}, denominator: {denominator[i, k]}")
                kld_loss = calc_kld(mu, log_var, reduction='none', beta=beta)
                train_kld_losses.extend(kld_loss.tolist())
                total_loss = recon_loss_per_image.mean() + kld_loss.mean()
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            elif cf['model'] == 'vqvae':
                x_hat, quantized, mu, log_var, commitment_loss, codebook_loss = model(x)
                x_hat = crop_to_size(x_hat, target_size)
                recon_loss = calc_recon_loss(x_hat, x, loss_type=cf['recon_loss'], reduction='mean', 
                                            weight_mode=weight_mode, return_per_class=True, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                recon_loss_per_image = calc_recon_loss(x_hat, x, loss_type=cf['recon_loss'], reduction='none', 
                                                      weight_mode=weight_mode, return_per_class=False, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                train_recon_losses.extend(recon_loss_per_image.tolist())
                
                if cf['train_label'] and cf['recon_loss'] == 'dice_bce':
                    target = (x > 0).float()
                    prob = torch.sigmoid(x_hat)
                    intersection = (prob * target).sum(dim=(2,3,4))
                    denominator = prob.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator   # shape [B,1]
                    for i in range(b_size):
                        train_dice_focal_losses[0].append(recon_loss[i, 0].item())
                        train_dice_scores[0].append(dice_score[i, 0].item())
                        if dice_score[i, 0] < 0 or dice_score[i, 0] > 1:
                            print(f"Invalid Dice score for class {label_names[1]}: {dice_score[i, 0]}, intersection: {intersection[i, 0]}, denominator: {denominator[i, 0]}")

                if cf['train_label'] and cf['recon_loss'] in ['dice', 'dice_entropy', 'dice_focal']:
                    x_one_hot = F.one_hot(x.squeeze(1).long(), num_classes=x_hat.shape[1]).permute(0, 4, 1, 2, 3).float()
                    intersection = (x_hat * x_one_hot).sum(dim=(2, 3, 4))
                    denominator = (x_hat.sum(dim=(2, 3, 4)) + x_one_hot.sum(dim=(2, 3, 4))) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator
                    for i in range(b_size):
                        for k in range(cf['out_ch']):
                            train_dice_focal_losses[k].append(recon_loss[i, k].item())
                            train_dice_scores[k].append(dice_score[i, k].item())
                            if dice_score[i, k] < 0 or dice_score[i, k] > 1:
                                print(f"Invalid Dice score for class {label_names[k]}: {dice_score[i, k]}, intersection: {intersection[i, k]}, denominator: {denominator[i, k]}")
                total_loss = recon_loss_per_image.mean() + commitment_loss + codebook_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            elif cf['model'] == 'infovae':
                mmd_weight = cf.get('mmd_weight', 1.0)
                beta = cf.get('beta', 0.0)
                if cf['beta_cyc_annealing']:
                    beta = cf['kld_weight'] * min(1, (global_step % cyc_annealing_batchs) / increasing_batchs)
                
                x_hat, _, mu, log_var = model(x)
                x_hat = crop_to_size(x_hat, target_size)
                recon_loss = calc_recon_loss(x_hat, x, loss_type=cf['recon_loss'], reduction='mean', 
                                            weight_mode=weight_mode, return_per_class=True, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                recon_loss_per_image = calc_recon_loss(x_hat, x, loss_type=cf['recon_loss'], reduction='none', 
                                                      weight_mode=weight_mode, return_per_class=False, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                train_recon_losses.extend(recon_loss_per_image.tolist())
                
                if cf['train_label'] and cf['recon_loss'] == 'dice_bce':
                    target = (x > 0).float()
                    prob = torch.sigmoid(x_hat)
                    intersection = (prob * target).sum(dim=(2,3,4))
                    denominator = prob.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator   # shape [B,1]
                    for i in range(b_size):
                        train_dice_focal_losses[0].append(recon_loss[i, 0].item())
                        train_dice_scores[0].append(dice_score[i, 0].item())
                        if dice_score[i, 0] < 0 or dice_score[i, 0] > 1:
                            print(f"Invalid Dice score for class {label_names[1]}: {dice_score[i, 0]}, intersection: {intersection[i, 0]}, denominator: {denominator[i, 0]}")

                if cf['train_label'] and cf['recon_loss'] in ['dice', 'dice_entropy', 'dice_focal']:
                    x_one_hot = F.one_hot(x.squeeze(1).long(), num_classes=x_hat.shape[1]).permute(0, 4, 1, 2, 3).float()
                    intersection = (x_hat * x_one_hot).sum(dim=(2, 3, 4))
                    denominator = (x_hat.sum(dim=(2, 3, 4)) + x_one_hot.sum(dim=(2, 3, 4))) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator
                    for i in range(b_size):
                        for k in range(cf['out_ch']):
                            train_dice_focal_losses[k].append(recon_loss[i, k].item())
                            train_dice_scores[k].append(dice_score[i, k].item())
                            if dice_score[i, k] < 0 or dice_score[i, k] > 1:
                                print(f"Invalid Dice score for class {label_names[k]}: {dice_score[i, k]}, intersection: {intersection[i, k]}, denominator: {denominator[i, k]}")
                
                z = model.reparameterize(mu, log_var)
                z_prior = torch.randn_like(z)
                mmd_loss = calc_mmd(z, z_prior, kernel='rbf', sigma=cf.get('mmd_sigma', 1.0))
                kld_loss = calc_kld(mu, log_var, reduction='none', beta=beta)
                train_kld_losses.extend(kld_loss.tolist())
                total_loss = recon_loss_per_image.mean() + mmd_weight * mmd_loss + kld_loss.mean()
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            elif cf['model'] == 'hybrid_vae_unet':
                if cf['beta_cyc_annealing']:
                    beta = cf['kld_weight'] * min(1, (global_step % cyc_annealing_batchs) / increasing_batchs)
                else:
                    beta = cf['kld_weight']
                
                vae_out, unet_out, _, mu, log_var = model(x)
                vae_out = crop_to_size(vae_out, target_size)
                unet_out = crop_to_size(unet_out, target_size)
                x_hat = unet_out
                
                vae_recon_loss = calc_recon_loss(vae_out, x, loss_type=cf['recon_loss'], reduction='mean', 
                                                weight_mode=weight_mode, return_per_class=True, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                unet_recon_loss = calc_recon_loss(unet_out, x, loss_type=cf['recon_loss'], reduction='mean', 
                                                 weight_mode=weight_mode, return_per_class=True, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                recon_loss_per_image = calc_recon_loss(unet_out, x, loss_type=cf['recon_loss'], reduction='none', 
                                                      weight_mode=weight_mode, return_per_class=False, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                train_recon_losses.extend(recon_loss_per_image.tolist())
                
                if cf['train_label'] and cf['recon_loss'] == 'dice_bce':
                    target = (x > 0).float()
                    prob = torch.sigmoid(x_hat)
                    intersection = (prob * target).sum(dim=(2,3,4))
                    denominator = prob.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator   # shape [B,1]
                    for i in range(b_size):
                        train_dice_focal_losses[0].append(recon_loss[i, 0].item())
                        train_dice_scores[0].append(dice_score[i, 0].item())
                        if dice_score[i, 0] < 0 or dice_score[i, 0] > 1:
                            print(f"Invalid Dice score for class {label_names[1]}: {dice_score[i, 0]}, intersection: {intersection[i, 0]}, denominator: {denominator[i, 0]}")

                if cf['train_label'] and cf['recon_loss'] in ['dice', 'dice_entropy', 'dice_focal']:
                    x_one_hot = F.one_hot(x.squeeze(1).long(), num_classes=unet_out.shape[1]).permute(0, 4, 1, 2, 3).float()
                    intersection = (unet_out * x_one_hot).sum(dim=(2, 3, 4))
                    denominator = (unet_out.sum(dim=(2, 3, 4)) + x_one_hot.sum(dim=(2, 3, 4))) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator
                    for i in range(b_size):
                        for k in range(cf['out_ch']):
                            train_dice_focal_losses[k].append(unet_recon_loss[i, k].item())
                            train_dice_scores[k].append(dice_score[i, k].item())
                            if dice_score[i, k] < 0 or dice_score[i, k] > 1:
                                print(f"Invalid Dice score for class {label_names[k]}: {dice_score[i, k]}, intersection: {intersection[i, k]}, denominator: {denominator[i, k]}")
                kld_loss = calc_kld(mu, log_var, reduction='none', beta=beta)
                train_kld_losses.extend(kld_loss.tolist())
                total_loss = vae_recon_loss.mean() + unet_recon_loss.mean() + kld_loss.mean()
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            elif cf['model'] in ['vae', 'paired_vae_label', 'vae_cbam']:
                if cf['beta_cyc_annealing']:
                    beta = cf['kld_weight'] * min(1, (global_step % cyc_annealing_batchs) / increasing_batchs)
                else:
                    beta = cf['kld_weight']

                if cf['model'] == 'vae_cbam':
                    x_hat, _, mu, log_var = model(x)
                else:
                    x_hat, _, mu, log_var = model(x)
                x_hat = crop_to_size(x_hat, target_size)
                print(f"x_hat shape: {x_hat.shape}, min: {x_hat.min().item()}, max: {x_hat.max().item()}")
                channel_sum = x_hat.sum(dim=1)
                print(f"Channel sum: min: {channel_sum.min()}, max: {channel_sum.max()}, mean: {channel_sum.mean()}")
                print(f"mu shape: {mu.shape}, log_var shape: {log_var.shape}")
                
                if cf['model'] == 'paired_vae_label':
                    assert cf['out_ch'] % 2 == 0, 'Paired VAE requires even number of output channels'
                    recon_loss = 0.0
                    for k in range(cf['out_ch'] // 2):
                        recon_loss_k = calc_recon_loss(x_hat[:, 2*k:2*(k+1)], x, loss_type=cf['recon_loss'], reduction='none', epoch = epoch + 1)
                        recon_loss_k = calc_recon_loss(k.mean(dim=(1,2,3,4)), label_mask = label_mask)
                        train_recon_loss_classes[k].extend(recon_loss_k.tolist())
                        recon_loss += recon_loss_k.mean()
                    train_recon_losses.extend(recon_loss_k.tolist())
                else:
                    recon_loss = calc_recon_loss(x_hat, x, loss_type=cf['recon_loss'], reduction='mean', weight_mode=weight_mode, return_per_class=True, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                    print(f"VAE recon_loss shape: shape: {recon_loss.shape}")
                    if cf['train_label'] and cf['recon_loss'] == 'dice_bce':
                        target = (x > 0).float()
                        prob = torch.sigmoid(x_hat)
                        intersection = (prob * target).sum(dim=(2,3,4))
                        denominator = prob.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + 1e-6
                        dice_score = (2 * intersection + 1e-6) / denominator   # shape [B,1]
                        for i in range(b_size):
                            train_dice_focal_losses[0].append(recon_loss[i, 0].item())
                            train_dice_scores[0].append(dice_score[i, 0].item())
                            if dice_score[i, 0] < 0 or dice_score[i, 0] > 1:
                                print(f"Invalid Dice score for class {label_names[1]}: {dice_score[i, 0]}, intersection: {intersection[i, 0]}, denominator: {denominator[i, 0]}")

                    if cf['train_label'] and cf['recon_loss'] in ['dice', 'dice_entropy', 'dice_focal']:
                        x_one_hot = F.one_hot(x.squeeze(1).long(), num_classes=x_hat.shape[1]).permute(0, 4, 1, 2, 3).float()
                        intersection = (x_hat * x_one_hot).sum(dim=(2, 3, 4))
                        denominator = (x_hat.sum(dim=(2, 3, 4)) + x_one_hot.sum(dim=(2, 3, 4))) + 1e-6
                        dice_score = (2 * intersection + 1e-6) / denominator
                        print(f"VAE unweighted dice_score shape: {dice_score.shape}")
                        for i in range(b_size):
                            for k in range(cf['out_ch']):
                                train_dice_focal_losses[k].append(recon_loss[i, k].item())
                                train_dice_scores[k].append(dice_score[i, k].item())
                                if dice_score[i, k] < 0 or dice_score[i, k] > 1:
                                    print(f"Dice score for class {label_names[k]}: {dice_score[i, k]}, intersection: {intersection[i, k]}, denominator: {denominator[i, k]}")
                    recon_loss_per_image = calc_recon_loss(x_hat, x, loss_type=cf['recon_loss'], reduction='none', weight_mode=weight_mode, return_per_class=False, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                    print(f"Per_image shape: {recon_loss_per_image.shape}")
                    train_recon_losses.extend(recon_loss_per_image.tolist())
                
                kld_loss = calc_kld(mu, log_var, reduction='none')
                print(f"KLD shape: {kld_loss.shape}")
                train_kld_losses.extend(kld_loss.tolist())
                total_loss = recon_loss_per_image.mean() + beta * kld_loss.mean()
                print(f"Total_loss shape: {total_loss.shape}, value: {total_loss.item()}")
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            elif cf['model'] == 'GMVAE2_3D':
                if cf['beta_cyc_annealing']:
                    beta = cf['kld_weight'] * min(1, (global_step % cyc_annealing_batchs) / increasing_batchs)
                else:
                    beta = cf['kld_weight']
                x_hat, z, mu, log_var, logits_y, qy, y_idx = model(x)
                x_hat = crop_to_size(x_hat, target_size)
                recon_loss = calc_recon_loss(x_hat, x, loss_type=cf['recon_loss'], reduction='mean', weight_mode=weight_mode, return_per_class=True, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                recon_loss_per_image = calc_recon_loss(x_hat, x, loss_type=cf['recon_loss'], reduction='none', weight_mode=weight_mode, return_per_class=False, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                train_recon_losses.extend(recon_loss_per_image.tolist())
                print(f"GMVAE recon_loss shape: {recon_loss.shape}")
                if cf['train_label'] and cf['recon_loss'] == 'dice_bce':
                    target = (x > 0).float()
                    prob = torch.sigmoid(x_hat)
                    intersection = (prob * target).sum(dim=(2,3,4))
                    denominator = prob.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator   # shape [B,1]
                    for i in range(b_size):
                        train_dice_focal_losses[0].append(recon_loss[i, 0].item())
                        train_dice_scores[0].append(dice_score[i, 0].item())
                        if dice_score[i, 0] < 0 or dice_score[i, 0] > 1:
                            print(f"Invalid Dice score for class {label_names[1]}: {dice_score[i, 0]}, intersection: {intersection[i, 0]}, denominator: {denominator[i, 0]}")

                if cf['train_label'] and cf['recon_loss'] in ['dice', 'dice_entropy', 'dice_focal']:
                    x_one_hot = F.one_hot(x.squeeze(1).long(), num_classes=x_hat.shape[1]).permute(0, 4, 1, 2, 3).float()
                    intersection = (x_hat * x_one_hot).sum(dim=(2, 3, 4))
                    denominator = (x_hat.sum(dim=(2, 3, 4)) + x_one_hot.sum(dim=(2, 3, 4))) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator
                    print(f"GMVAE unweighted dice_score shape: {dice_score.shape}")
                    for i in range(b_size):
                        for k in range(cf['out_ch']):
                            train_dice_focal_losses[k].append(recon_loss[i, k].item())
                            train_dice_scores[k].append(dice_score[i, k].item())
                            if dice_score[i, k] < 0 or dice_score[i, k] > 1:
                                print(f"Dice score for class {label_names[k]}: {dice_score[i, k]}, intersection: {intersection[i, k]}, denominator: {denominator[i, k]}")
                recon_loss_per_image = calc_recon_loss(x_hat, x, loss_type=cf['recon_loss'], reduction='none', weight_mode=weight_mode, return_per_class=False, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                print(f"Per_image shape: {recon_loss_per_image.shape}")
                train_recon_losses.extend(recon_loss_per_image.tolist())
                zm_prior = model.zm_prior
                zv_prior = model.zv_prior
                print(f"zm_prior: {zm_prior}, zv_prior: {zv_prior}")
                print(f"zm_prior[0] shape: {zm_prior[0].shape}, zv_prior[0] shape: {zv_prior[0].shape}")
                kld_loss = gmvae_kld_loss(mu, log_var, qy, zm_prior, zv_prior, reduction='none')
                print(f"KLD shape: {kld_loss.shape}")
                train_kld_losses.extend(kld_loss.tolist())
                total_loss = recon_loss_per_image.mean() + beta * kld_loss.mean()
                print(f"Total_loss shape: {total_loss.shape}, value: {total_loss.item()}")
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                
            elif cf['model'] in ['intro_vae', 'soft_intro_vae', 'reverse_vae']:
                noise_batch = torch.randn(size=(b_size, cf['z_dim'])).to(device=device)
                real_batch = x

                for param in model.encoder.parameters():
                    param.requires_grad = True
                for param in model.decoder.parameters():
                    param.requires_grad = False

                fake = model.decode(noise_batch)
                fake = crop_to_size(fake, target_size)
                if cf['model'] == 'reverse_vae':
                    real_mu, real_log_var, anomaly_embeddings = model.encode(real_batch)
                else:
                    real_mu, real_log_var = model.encode(real_batch)
                z = model.reparameterize(real_mu, real_log_var)
                rec = model.decoder(z)
                rec = crop_to_size(rec, target_size)

                recon_loss = calc_recon_loss(rec, real_batch, loss_type=cf['recon_loss'], reduction='mean', weight_mode=weight_mode, return_per_class=True, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                print(f"IntroVAE recon_loss shape: {recon_loss.shape}")
                recon_loss_per_image = calc_recon_loss(rec, real_batch, loss_type=cf['recon_loss'], reduction='none', weight_mode=weight_mode, return_per_class=False, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                print(f"Recon_loss_per_image shape: {recon_loss_per_image.shape}")
                train_recon_losses.extend(recon_loss_per_image.tolist())
                if cf['train_label'] and cf['recon_loss'] == 'dice_bce':
                    target = (x > 0).float()
                    prob = torch.sigmoid(x_hat)
                    intersection = (prob * target).sum(dim=(2,3,4))
                    denominator = prob.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator   # shape [B,1]
                    for i in range(b_size):
                        train_dice_focal_losses[0].append(recon_loss[i, 0].item())
                        train_dice_scores[0].append(dice_score[i, 0].item())
                        if dice_score[i, 0] < 0 or dice_score[i, 0] > 1:
                            print(f"Invalid Dice score for class {label_names[1]}: {dice_score[i, 0]}, intersection: {intersection[i, 0]}, denominator: {denominator[i, 0]}")

                if cf['train_label'] and cf['recon_loss'] in ['dice', 'dice_entropy', 'dice_focal']:
                    x_one_hot = F.one_hot(real_batch.squeeze(1).long(), num_classes=rec.shape[1]).permute(0, 4, 1, 2, 3).float()
                    intersection = (rec * x_one_hot).sum(dim=(2, 3, 4))
                    denominator = (rec.sum(dim=(2, 3, 4)) + x_one_hot.sum(dim=(2, 3, 4))) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator
                    print(f"IntroVAE unweighted dice_score shape: {dice_score.shape}")
                    for i in range(b_size):
                        for k in range(dice_score.shape[1]):
                            train_dice_focal_losses[k].append(recon_loss[i, k].item())
                            train_dice_scores[k].append(dice_score[i, k].item())
                            if dice_score[i, k] < 0 or dice_score[i, k] > 1:
                                print(f"Invalid Dice score for class {label_names[k]}: {dice_score[i, k]}, intersection: {intersection[i, k]}, denominator: {denominator[i, k]}")

                lossE_real_kld = calc_kld(real_mu, real_log_var, reduction='none')
                train_kld_losses.extend(lossE_real_kld.tolist())
                
                rec_labels = torch.argmax(rec, dim=1, keepdim=True).float()
                rec_labels = crop_to_size(rec_labels, target_size)
                fake_labels = torch.argmax(fake, dim=1, keepdim=True).float()
                fake_labels = crop_to_size(fake_labels, target_size)
                
                print(f"rec shape: {rec.shape}, rec_labels shape: {rec_labels.shape}")
                print(f"fake shape: {fake.shape}, fake_labels shape: {fake_labels.shape}")

                if cf['model'] == 'reverse_vae':
                    rec_rec, z_rec, rec_mu, rec_log_var, healthy_embeddings = model(rec_labels.detach())
                    rec_fake, z_fake, fake_mu, fake_log_var, _ = model(fake_labels.detach())
                else:
                    rec_rec, z_rec, rec_mu, rec_log_var = model(rec_labels.detach())
                    rec_fake, z_fake, fake_mu, fake_log_var = model(fake_labels.detach())

                rec_rec = crop_to_size(rec_rec, target_size)
                rec_fake = crop_to_size(rec_fake, target_size)
                print(f"rec_rec shape: {rec_rec.shape}, rec_fake shape: {rec_fake.shape}")
                
                kld_rec = calc_kld(rec_mu, rec_log_var, reduction='none')
                kld_fake = calc_kld(fake_mu, fake_log_var, reduction='none')

                print(f"Computing loss_rec_rec_e with loss_type: {cf['recon_loss']}, weight_mode: {weight_mode}")
                loss_rec_rec_e = calc_recon_loss(rec_rec, rec_labels, loss_type=cf['recon_loss'], reduction='none', weight_mode=weight_mode, return_per_class=True, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                # Handle list return (for some loss types)
                if isinstance(loss_rec_rec_e, list):
                    loss_rec_rec_e = loss_rec_rec_e[0]
                print(f"loss_rec_rec_e shape before mean: {loss_rec_rec_e.shape}")
                if loss_rec_rec_e.dim() == 2:  # Shape [batch_size, num_classes]
                    loss_rec_rec_e = loss_rec_rec_e.mean(dim=1)  # Mean over classes
                elif loss_rec_rec_e.dim() == 5:  # Shape [batch, channels, D, H, W] for gaussian_nll
                    loss_rec_rec_e = loss_rec_rec_e.mean(dim=(1, 2, 3, 4))  # Mean over channels and spatial dims
                else:
                    raise ValueError(f"Unexpected loss_rec_rec_e shape: {loss_rec_rec_e.shape}")
                print(f"loss_rec_rec_e shape after mean: {loss_rec_rec_e.shape}")
                
                print(f"Computing loss_rec_fake_e with loss_type: {cf['recon_loss']}, weight_mode: {weight_mode}")
                loss_rec_fake_e = calc_recon_loss(rec_fake, fake_labels, loss_type=cf['recon_loss'], reduction='none', weight_mode=weight_mode, return_per_class=True, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                # Handle list return (for some loss types)
                if isinstance(loss_rec_fake_e, list):
                    loss_rec_fake_e = loss_rec_fake_e[0]
                print(f"loss_rec_fake_e shape before mean: {loss_rec_fake_e.shape}")
                if loss_rec_fake_e.dim() == 2:  # Shape [batch_size, num_classes]
                    loss_rec_fake_e = loss_rec_fake_e.mean(dim=1)  # Mean over classes
                elif loss_rec_fake_e.dim() == 5:  # Shape [batch, channels, D, H, W] for gaussian_nll
                    loss_rec_fake_e = loss_rec_fake_e.mean(dim=(1, 2, 3, 4))  # Mean over channels and spatial dims
                else:
                    raise ValueError(f"Unexpected loss_rec_fake_e shape: {loss_rec_fake_e.shape}")
                print(f"loss_rec_fake_e shape after mean: {loss_rec_fake_e.shape}")

                input_dim = [cf['in_ch']] + cf['ori_size']
                scale = 1 / np.prod(input_dim)
                expelbo_rec = (-2 * scale * (siv_cf['beta_rec'] * loss_rec_rec_e + siv_cf['beta_neg'] * kld_rec)).exp().mean()
                expelbo_fake = (-2 * scale * (siv_cf['beta_rec'] * loss_rec_fake_e + siv_cf['beta_neg'] * kld_fake)).exp().mean()

                if cf['model'] == 'intro_vae':
                    lossE_margin = F.relu(siv_cf['m_plus'] - calc_kld(rec_mu, rec_log_var, reduction='mean')) + \
                                   F.relu(siv_cf['m_plus'] - calc_kld(fake_mu, fake_log_var, reduction='mean'))
                    lossE = recon_loss_per_image.mean() * siv_cf['beta_rec'] + lossE_real_kld.mean() * siv_cf['beta_kld'] + lossE_margin * siv_cf['beta_neg']
                else:
                    lossE_real = scale * (siv_cf['beta_rec'] * recon_loss_per_image.mean() + siv_cf['beta_kld'] * lossE_real_kld.mean())
                    lossE_fake = 0.25 * (expelbo_rec + expelbo_fake)
                    lossE = lossE_real + 0.25 * (expelbo_rec + expelbo_fake)
                    if cf['model'] == 'reverse_vae':
                        loss_emb = embedding_loss(anomaly_embeddings, healthy_embeddings)
                        lossE += siv_cf['embed_weight'] * loss_emb

                optimizer_e.zero_grad()
                lossE.backward()
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1.0)
                optimizer_e.step()

                for param in model.encoder.parameters():
                    param.requires_grad = False
                for param in model.decoder.parameters():
                    param.requires_grad = True

                fake = model.decode(noise_batch)
                fake = crop_to_size(fake, target_size)
                rec = model.decoder(z.detach())
                rec = crop_to_size(rec, target_size)
                recon_loss = calc_recon_loss(rec, real_batch, loss_type=cf['recon_loss'], reduction='mean', weight_mode=weight_mode, return_per_class=True, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                print(f"IntroVAE decoder recon_loss shape: {recon_loss.shape}")
                recon_loss_per_image = calc_recon_loss(rec, real_batch, loss_type=cf['recon_loss'], reduction='none', weight_mode=weight_mode, return_per_class=False, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                print(f"recon_loss_per_image shape: {recon_loss_per_image.shape}")
                train_recon_losses.extend(recon_loss_per_image.tolist())
                if cf['train_label'] and cf['recon_loss'] == 'dice_bce':
                    target = (x > 0).float()
                    prob = torch.sigmoid(rec)
                    intersection = (prob * target).sum(dim=(2,3,4))
                    denominator = prob.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator   # shape [B,1]
                    for i in range(b_size):
                        train_dice_focal_losses[0].append(recon_loss[i, 0].item())
                        train_dice_scores[0].append(dice_score[i, 0].item())
                        if dice_score[i, 0] < 0 or dice_score[i, 0] > 1:
                            print(f"Invalid Dice score for class {label_names[1]}: {dice_score[i, 0]}, intersection: {intersection[i, 0]}, denominator: {denominator[i, 0]}")

                if cf['train_label'] and cf['recon_loss'] in ['dice', 'dice_entropy', 'dice_focal']:
                    x_one_hot = F.one_hot(real_batch.squeeze(1).long(), num_classes=rec.shape[1]).permute(0, 4, 1, 2, 3).float()
                    intersection = (rec * x_one_hot).sum(dim=(2, 3, 4))
                    denominator = (rec.sum(dim=(2, 3, 4)) + x_one_hot.sum(dim=(2, 3, 4))) + 1e-6
                    dice_score = (2 * intersection + 1e-6) / denominator
                    print(f"IntroVAE decoder unweighted dice_score shape: {dice_score.shape}")
                    for i in range(b_size):
                        for k in range(dice_score.shape[1]):
                            train_dice_focal_losses[k].append(recon_loss[i, k].item())
                            train_dice_scores[k].append(dice_score[i, k].item())
                            if dice_score[i, k] < 0 or dice_score[i, k] > 1:
                                print(f"Invalid Dice score for class {label_names[k]}: {dice_score[i, k]}, intersection: {intersection[i, k]}, denominator: {denominator[i, k]}")

                if cf['model'] == 'reverse_vae':
                    rec_mu, rec_log_var, _ = model.encode(rec)
                    fake_mu, fake_log_var, _ = model.encode(fake)
                else:
                    if cf['train_label']:
                        rec_mu, rec_log_var = model.encode(rec_labels)
                        fake_mu, fake_log_var = model.encode(fake_labels)
                    else:
                        rec_mu, rec_log_var = model.encode(rec)
                        fake_mu, fake_log_var = model.encode(fake)

                z_rec = model.reparameterize(rec_mu, rec_log_var)
                z_fake = model.reparameterize(fake_mu, fake_log_var)
                rec_rec = model.decoder(z_rec)
                rec_fake = model.decoder(z_fake)
                rec_rec = crop_to_size(rec_rec, target_size)
                rec_fake = crop_to_size(rec_fake, target_size)

                loss_rec_rec = calc_recon_loss(rec_rec, rec.detach(), loss_type=cf['recon_loss'], reduction='batch_mean', weight_mode=weight_mode, return_per_class=False, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)
                loss_fake_rec = calc_recon_loss(rec_fake, fake.detach(), loss_type=cf['recon_loss'], reduction='batch_mean', weight_mode=weight_mode, return_per_class=False, alpha=cf.get('alpha', 0.5), epoch = epoch + 1)

                lossD_rec_kld = calc_kld(rec_mu, rec_log_var, reduction='mean')
                lossD_fake_kld = calc_kld(fake_mu, fake_log_var, reduction='mean')

                if cf['model'] == 'intro_vae':
                    lossD = (lossD_rec_kld + lossD_fake_kld) * siv_cf['beta_neg']
                else:
                    lossD_real = scale * siv_cf['beta_rec'] * recon_loss_per_image.mean()
                    lossD_fake = scale * 0.5 * (siv_cf['beta_kld'] * (lossD_rec_kld + lossD_fake_kld) +
                                         siv_cf['gamma_r'] * siv_cf['beta_rec'] * (loss_rec_rec + loss_fake_rec))
                    lossD = lossD_real + lossD_fake

                optimizer_d.zero_grad()
                lossD.backward()
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm=1.0)
                optimizer_d.step()
            train_image_count += b_size
            global_step += 1

            recon_loss_np = np.mean(train_recon_losses[-b_size:]) if train_recon_losses else 0.0
            kld_loss_np = kld_loss.mean().item() if 'kld_loss' in locals() else (lossE_real_kld.mean().item() if 'lossE_real_kld' in locals() else 0)
            total_loss_np = recon_loss_np + (kld_loss_np * cf.get('kld_weight', 0) if cf['model'] in ['vae', 'paired_vae_label'] else 0)
            pbar.set_postfix(recon_loss=recon_loss_np, kld_loss=kld_loss_np, total_loss=total_loss_np)

        train_recon_loss_avg = np.mean(train_recon_losses) if train_recon_losses else 0.0
        train_kld_loss_avg = np.mean(train_kld_losses) if train_kld_losses else 0.0
        train_dice_score_classes = np.array([np.mean(scores) if scores else 0.0 for scores in train_dice_scores])
        train_dice_focal_loss_classes = np.array([np.mean(losses) if losses else 0.0 for losses in train_dice_focal_losses])
        if cf['model'] == 'paired_vae_label':
            train_recon_loss_classes = np.array([np.mean(losses) if losses else 0.0 for losses in train_recon_loss_classes])

        print(f"\nTraining Epoch {epoch + 1}/{cf['num_epochs']}:")
        print(f"  Recon Loss: {train_recon_loss_avg:.6f}, KLD Loss: {train_kld_loss_avg:.6f}")
        if cf['train_label'] and cf['recon_loss'] == 'dice_bce':
            wandb.log({
                f'Training Dice-BCE Loss': train_dice_focal_loss_classes[0],
                f'Training Unweighted Dice Score - {label_names[1]}': train_dice_score_classes[0]
            },step = epoch + 1)
            print(f"  Training Dice-BCE Loss: {train_dice_focal_loss_classes[0]:.6f}")
            print(f"  Training Unweighted Dice Score for {label_names[1]}: {train_dice_score_classes[0]:.6f}")
        if cf['train_label'] and cf['recon_loss'] in ['dice', 'dice_entropy', 'dice_focal']:
            for k in range(cf['out_ch']):
                wandb.log({
                    f'Training Dice-Focal Loss - {label_names[k]}': train_dice_focal_loss_classes[k],
                    f'Training Unweighted Dice Score - {label_names[k]}': train_dice_score_classes[k]
                },step = epoch + 1)
                print(f"  Training Dice-Focal Loss for {label_names[k]}: {train_dice_focal_loss_classes[k]:.6f}")
                print(f"  Training Unweighted Dice Score for {label_names[k]}: {train_dice_score_classes[k]:.6f}")
        if cf['model'] == 'paired_vae_label':
            for k in range(cf['out_ch'] // 2):
                wandb.log({f"Training gaussian nll loss for label {k}": train_recon_loss_classes[k]},step = epoch + 1)
                print(f"  Training gaussian nll loss for label {k}: {train_recon_loss_classes[k]:.6f}")

        tb_writer.add_scalar('Train/Epoch_Recon_loss', train_recon_loss_avg, epoch + 1)
        tb_writer.add_scalar('Train/Epoch_KLD_loss', train_kld_loss_avg, epoch + 1)
        for k in range(cf['out_ch']):
            tb_writer.add_scalar(f'Train/Dice_Focal_Loss_{label_names[k]}', train_dice_focal_loss_classes[k], epoch + 1)
            tb_writer.add_scalar(f'Train/Unweighted_Dice_Score_{label_names[k]}', train_dice_score_classes[k], epoch + 1)
        wandb.log({"Train Epoch Recon Loss": train_recon_loss_avg, "Train Epoch KLD Loss": train_kld_loss_avg},step = epoch + 1)

        if cf['model'] in ['intro_vae', 'soft_intro_vae', 'reverse_vae']:
            scheduler_e.step()
            scheduler_d.step()
            tb_writer.add_scalar('Learning_rate_e', optimizer_e.param_groups[0]['lr'], epoch + 1)
            tb_writer.add_scalar('Learning_rate_d', optimizer_d.param_groups[0]['lr'], epoch + 1)
        else:
            scheduler.step()
            tb_writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch + 1)
        x_train = x
        name_tr = xname[0]
        xhat_train = x_hat
        if cf['train_label']:
            x_train_axial, xhat_train_axial, diff_train_axial = make_slice_triplet(x, x_hat if 'x_hat' in locals() else rec, cf, axis='axial')
            x_train_sagittal, xhat_train_sagittal, diff_train_sagittal = make_slice_triplet(x, x_hat if 'x_hat' in locals() else rec, cf, axis='sagittal')
            x_train_coronal, xhat_train_coronal, diff_train_coronal = make_slice_triplet(x, x_hat if 'x_hat' in locals() else rec, cf, axis='coronal')
        else:
            x_train_axial, xhat_train_axial, diff_train_axial = make_slice_triplet(x, x_hat if 'x_hat' in locals() else rec, cf, axis='axial')
            x_train_sagittal, xhat_train_sagittal, diff_train_sagittal = make_slice_triplet(x, x_hat if 'x_hat' in locals() else rec, cf, axis='sagittal')
            x_train_coronal, xhat_train_coronal, diff_train_coronal = make_slice_triplet(x, x_hat if 'x_hat' in locals() else rec, cf, axis='coronal')
        print(f"x train axial slice shape: {x_train_axial.shape}, max: {x_train_axial.max().item()}, min: {x_train_axial.min().item()}")
        print(f"xhat train axial slice shape: {xhat_train_axial.shape}, max: {xhat_train_axial.max().item()}, min: {xhat_train_axial.min().item()}")
        print(f"diff train axial shape: {diff_train_axial.shape}, max: {diff_train_axial.max().item()}, min: {diff_train_axial.min().item()}")
        print(f"x train sagittal slice shape: {x_train_sagittal.shape}, max: {x_train_sagittal.max().item()}, min: {x_train_sagittal.min().item()}")
        print(f"xhat train sagittal slice shape: {xhat_train_sagittal.shape}, max: {xhat_train_sagittal.max().item()}, min: {xhat_train_sagittal.min().item()}")
        print(f"diff train sagittal shape: {diff_train_sagittal.shape}, max: {diff_train_sagittal.max().item()}, min: {diff_train_sagittal.min().item()}")
        print(f"x train coronal slice shape: {x_train_coronal.shape}, max: {x_train_coronal.max().item()}, min: {x_train_coronal.min().item()}")
        print(f"xhat train coronal slice shape: {xhat_train_coronal.shape}, max: {xhat_train_coronal.max().item()}, min: {xhat_train_coronal.min().item()}")
        print(f"diff train coronal shape: {diff_train_coronal.shape}, max: {diff_train_coronal.max().item()}, min: {diff_train_coronal.min().item()}")
        if n_val_images > 0:
            model.eval()
            val_recon_losses = []
            val_kld_losses = []
            val_dice_scores = [[] for _ in range(cf['out_ch'])]
            val_dice_focal_losses = [[] for _ in range(cf['out_ch'])]
            val_commitment_losses = []
            val_codebook_losses = []
            val_image_count = 0

            if cf['model'] == 'paired_vae_label':
                val_recon_loss_classes = [[] for _ in range(cf['out_ch'] // 2)]
            else:
                val_recon_loss_classes = None

            for val_input_proj in val_loader:
                x_in = val_input_proj['proj_image'].to(device)
                x_target = val_input_proj.get('target_image', val_input_proj['proj_image']).to(device)
                xname = val_input_proj['proj_name']

                x_in = NormalizeImage(x_in, cf)

                # If Post-DAE is active, x_in is already binary {0,1}
                if cf['pre_processing'].get('post_dae_train', False) and cf.get('train', False):
                    x_in = (x_in > 0).float()
                else:
                    x_in = apply_label_mask(x_in, cf)

                x_target = NormalizeImage(x_target, cf)
                x_target = apply_label_mask(x_target, cf)

                b_size = x_in.shape[0]
                with torch.no_grad():
                    if cf['model'] == 'reverse_vae':
                        x_hat, _, mu, log_var, _ = model(x_in)
                    elif cf['model'] == 'cvae':
                        labels = x_in.squeeze(1).long().view(b_size, -1)
                        cond = torch.zeros(b_size, cf['out_ch'], device=device)
                        for i in range(b_size):
                            counts = torch.bincount(labels[i], minlength=cf['out_ch']).float()
                            cond[i] = counts / counts.sum()
                        cond = cond.to(device)
                        x_hat, _, mu, log_var = model(x_in, cond)
                    elif cf['model'] == 'vqvae':
                        x_hat, _, mu, log_var, commitment_loss, codebook_loss = model(x_in)
                        val_commitment_losses.append(commitment_loss.item())
                        val_codebook_losses.append(codebook_loss.item())
                    elif cf['model'] == 'hybrid_vae_unet':
                        vae_out, unet_out, _, mu, log_var = model(x_in)
                        x_hat = unet_out
                    elif cf['model'] == 'GMVAE2_3D':
                        x_hat, _, mu, log_var, logits_y, qy, y_idx = model(x_in)
                    elif cf['model'] == 'ae':
                        x_hat, _, _, _ = model(x_in, deterministic=True)
                    else:
                        x_hat, _, mu, log_var = model(x_in)
                    x_hat = crop_to_size(x_hat, target_size)
                    if cf['model'] == 'paired_vae_label':
                        recon_loss = 0.0
                        for k in range(cf['out_ch'] // 2):
                            recon_loss_k = calc_recon_loss(
                                x_hat[:, 2*k:2*(k+1)],
                                x_target,
                                loss_type=cf['recon_loss'],
                                reduction='none',
                                epoch=epoch + 1
                            )
                            recon_loss_k = recon_loss_k.mean(dim=(1, 2, 3, 4))
                            val_recon_loss_classes[k].extend(recon_loss_k.tolist())
                            val_recon_losses.extend(recon_loss_k.tolist())
                    else:
                        recon_loss = calc_recon_loss(
                            x_hat,
                            x_target,
                            loss_type=cf['recon_loss'],
                            reduction='mean',
                            weight_mode=weight_mode,
                            return_per_class=True,
                            alpha=cf.get('alpha', 0.5),
                            epoch=epoch + 1
                        )

                        if cf['train_label'] and cf['recon_loss'] == 'dice_bce':
                            target = (x_target > 0).float()
                            prob = torch.sigmoid(x_hat)
                            intersection = (prob * target).sum(dim=(2, 3, 4))
                            denominator = prob.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) + 1e-6
                            dice_score = (2 * intersection + 1e-6) / denominator  # [B,1]

                            for i in range(b_size):
                                val_dice_focal_losses[0].append(recon_loss[i, 0].item())
                                val_dice_scores[0].append(dice_score[i, 0].item())
                                if dice_score[i, 0] < 0 or dice_score[i, 0] > 1:
                                    print(
                                        f"Val Invalid Dice score for class {label_names[1]}: "
                                        f"{dice_score[i, 0]}, intersection: {intersection[i, 0]}, denominator: {denominator[i, 0]}"
                                    )

                        elif cf['train_label'] and cf['recon_loss'] in ['dice', 'dice_entropy', 'dice_focal']:
                            x_one_hot = F.one_hot(
                                x_target.squeeze(1).long(),
                                num_classes=x_hat.shape[1]
                            ).permute(0, 4, 1, 2, 3).float()

                            intersection = (x_hat * x_one_hot).sum(dim=(2, 3, 4))
                            denominator = (x_hat.sum(dim=(2, 3, 4)) + x_one_hot.sum(dim=(2, 3, 4))) + 1e-6
                            dice_score = (2 * intersection + 1e-6) / denominator

                            for i in range(b_size):
                                for k in range(dice_score.shape[1]):
                                    val_dice_focal_losses[k].append(recon_loss[i, k].item())
                                    val_dice_scores[k].append(dice_score[i, k].item())
                                    if dice_score[i, k] < 0 or dice_score[i, k] > 1:
                                        print(
                                            f"Val Invalid Dice score for class {label_names[k]}: "
                                            f"{dice_score[i, k]}, intersection: {intersection[i, k]}, denominator: {denominator[i, k]}"
                                        )

                        recon_loss_per_image = calc_recon_loss(
                            x_hat,
                            x_target,
                            loss_type=cf['recon_loss'],
                            reduction='none',
                            weight_mode=weight_mode,
                            return_per_class=False,
                            alpha=cf.get('alpha', 0.5),
                            epoch=epoch + 1
                        )

                        val_recon_losses.extend(recon_loss_per_image.tolist())
                    if cf['model'] != 'ae' and cf['model'] != 'vqvae':
                        val_kld_loss = calc_kld(mu, log_var, reduction='none')
                        print(f"Val_kld_loss shape: {val_kld_loss.shape}")
                        val_kld_losses.extend(val_kld_loss.tolist())
                    val_image_count += b_size
                x_val = x_target
                print(f"xname valid is {xname}")
                name_va = xname[0]
                xhat_val = x_hat
                if cf['train_label']:
                    x_valid_axial, xhat_valid_axial, diff_valid_axial = make_slice_triplet(x_target, x_hat if 'x_hat' in locals() else rec, cf, axis='axial')
                    x_valid_sagittal, xhat_valid_sagittal, diff_valid_sagittal = make_slice_triplet(x_target, x_hat if 'x_hat' in locals() else rec, cf, axis='sagittal')
                    x_valid_coronal, xhat_valid_coronal, diff_valid_coronal = make_slice_triplet(x_target, x_hat if 'x_hat' in locals() else rec, cf, axis='coronal')
                else:
                    x_valid_axial, xhat_valid_axial, diff_valid_axial = make_slice_triplet(x_target, x_hat if 'x_hat' in locals() else rec, cf, axis='axial')
                    x_valid_sagittal, xhat_valid_sagittal, diff_valid_sagittal = make_slice_triplet(x_target, x_hat if 'x_hat' in locals() else rec, cf, axis='sagittal')
                    x_valid_coronal, xhat_valid_coronal, diff_valid_coronal = make_slice_triplet(x_target, x_hat if 'x_hat' in locals() else rec, cf, axis='coronal')

            print(f"x valid axial slice shape: {x_valid_axial.shape}, max: {x_valid_axial.max().item()}, min: {x_valid_axial.min().item()}")
            print(f"xhat valid axial slice shape: {xhat_valid_axial.shape}, max: {xhat_valid_axial.max().item()}, min: {xhat_valid_axial.min().item()}")
            print(f"diff valid axial shape: {diff_valid_axial.shape}, max: {diff_valid_axial.max().item()}, min: {diff_valid_axial.min().item()}")
            print(f"x valid sagittal slice shape: {x_valid_sagittal.shape}, max: {x_valid_sagittal.max().item()}, min: {x_valid_sagittal.min().item()}")
            print(f"xhat valid sagittal slice shape: {xhat_valid_sagittal.shape}, max: {xhat_valid_sagittal.max().item()}, min: {xhat_valid_sagittal.min().item()}")
            print(f"diff valid sagittal shape: {diff_valid_sagittal.shape}, max: {diff_valid_sagittal.max().item()}, min: {diff_valid_sagittal.min().item()}")
            print(f"x valid coronal slice shape: {x_valid_coronal.shape}, max: {x_valid_coronal.max().item()}, min: {x_valid_coronal.min().item()}")
            print(f"xhat valid coronal slice shape: {xhat_valid_coronal.shape}, max: {xhat_valid_coronal.max().item()}, min: {xhat_valid_coronal.min().item()}")
            print(f"diff valid coronal shape: {diff_valid_coronal.shape}, max: {diff_valid_coronal.max().item()}, min: {diff_valid_coronal.min().item()}")
            val_recon_loss_avg = np.mean(val_recon_losses) if val_recon_losses else 0.0
            val_kld_loss_avg = np.mean(val_kld_losses) if val_kld_losses else 0.0
            val_dice_score_classes = np.array([np.mean(scores) if scores else 0.0 for scores in val_dice_scores])
            val_dice_focal_loss_classes = np.array([np.mean(losses) if losses else 0.0 for losses in val_dice_focal_losses])
            val_commitment_loss_avg = np.mean(val_commitment_losses) if val_commitment_losses else 0.0
            val_codebook_loss_avg = np.mean(val_codebook_losses) if val_codebook_losses else 0.0

            if cf['model'] == 'vqvae':
                val_total_loss = val_recon_loss_avg + val_commitment_loss_avg + val_codebook_loss_avg
            else:
                val_total_loss = val_recon_loss_avg + (val_kld_loss_avg * cf.get('kld_weight', 0) if cf['model'] in ['vae', 'paired_vae_label'] else 0)

            print(f"\nValidation Epoch {epoch + 1} Average (over {val_image_count} images):")
            print(f"  Recon Loss: {val_recon_loss_avg:.6f}, KLD: {val_kld_loss_avg:.6f}, Total Loss: {val_total_loss:.6f}")
            if cf['train_label'] and cf['recon_loss'] == 'dice_bce':
                wandb.log({
                    f'Validation Dice-BCE Loss': val_dice_focal_loss_classes[0],
                    f'Validation Unweighted Dice Score - {label_names[1]}': val_dice_score_classes[0]
                },step = epoch + 1)
                print(f"  Validation Dice-BCE Loss: {val_dice_focal_loss_classes[0]:.6f}")
                print(f"  Validation Unweighted Dice Score for {label_names[1]}: {val_dice_score_classes[0]:.6f}")
            if cf['train_label'] and cf['recon_loss'] in ['dice', 'dice_entropy', 'dice_focal']:
                for k in range(cf['out_ch']):
                    wandb.log({
                        f'Validation Dice-Focal Loss - {label_names[k]}': val_dice_focal_loss_classes[k],
                        f'Validation Unweighted Dice Score - {label_names[k]}': val_dice_score_classes[k]
                    },step = epoch + 1)
                    print(f"  Validation Dice-Focal Loss for {label_names[k]}: {val_dice_focal_loss_classes[k]:.6f}")
                    print(f"  Validation Unweighted Dice Score for {label_names[k]}: {val_dice_score_classes[k]:.6f}")

            tb_writer.add_scalar('Validation/Epoch_Recon_loss', val_recon_loss_avg, epoch + 1)
            tb_writer.add_scalar('Validation/Epoch_KLD_loss', val_kld_loss_avg, epoch + 1)
            tb_writer.add_scalar('Validation/Epoch_Total_loss', val_total_loss, epoch + 1)
            for k in range(cf['out_ch']):
                tb_writer.add_scalar(f'Validation/Dice_Focal_Loss_{label_names[k]}', val_dice_focal_loss_classes[k], epoch + 1)
                tb_writer.add_scalar(f'Validation/Unweighted_Dice_Score_{label_names[k]}', val_dice_score_classes[k], epoch + 1)
            wandb.log({
                "Validation Epoch Recon Loss": val_recon_loss_avg,
                "Validation Epoch KLD Loss": val_kld_loss_avg
            },step = epoch + 1)
            if cf['model'] == 'vqvae':
                wandb.log({
                    "Validation Commitment Loss": val_commitment_loss_avg,
                    "Validation Codebook Loss": val_codebook_loss_avg
                },step = epoch + 1)

        if (epoch + 1) % cf['save_interval'] == 0:
            with torch.no_grad():
                if cf['model'] == 'cvae':
                    b_size = 1
                    labels = x[0].squeeze(1).long().view(b_size, -1)
                    cond = torch.zeros(b_size, cf['out_ch'], device=device)
                    for i in range(b_size):
                        counts = torch.bincount(labels[i], minlength=cf['out_ch']).float()
                        cond[i] = counts / counts.sum()
                    cond = cond.to(device)
                    # x_hat, _, mu, log_var = model(x, cond)
                    sample_recon = model.sample(1, device, cond)[0]
                elif cf['model'] == 'vqvae':
                    x_hat, _, mu, _, _, _ = model(x)
                    sample_recon = model.sample(1, device)[0]
                elif cf['model'] == 'hybrid_vae_unet':
                    vae_out, unet_out, _, mu, log_var = model(x)
                    x_hat = unet_out
                    sample_recon = model.sample(1, device)[1]
                elif cf['model'] == 'GMVAE2_3D':
                    x_hat, _, mu, log_var, logits_y, qy, y_idx = model(x)
                    sample_recon = model.sample(1, device)[0]
                else:
                    x_hat, _, mu, log_var = model(x)
                    sample_recon = model.sample(1, device)[0]
                sample_recon = crop_to_size(sample_recon, target_size)
                print(f"sample recon shape is {sample_recon.shape}")
            if cf['train_label']:
                sample_recon = compute_slice_from_img(sample_recon[0], cf, mode='prob')
            else:
                sample_recon = compute_slice_from_img(sample_recon[0], cf, mode=None)
            print(f"sample recon slice shape is {sample_recon.shape}")
            # tb_writer.add_images('Validation/Sample_recon', sample_recon, epoch + 1, dataformats='HW')
            current_table = wandb.Table(columns=my_table.columns, data=my_table.data)
            current_table.add_data(f"Epoch {epoch} Axial", wandb.Image(xhat_train_axial), wandb.Image(x_train_axial), wandb.Image(diff_train_axial), wandb.Image(xhat_valid_axial), wandb.Image(x_valid_axial), wandb.Image(diff_valid_axial), wandb.Image(sample_recon))
            current_table.add_data(f"Epoch {epoch} Sagittal", wandb.Image(xhat_train_sagittal), wandb.Image(x_train_sagittal), wandb.Image(diff_train_sagittal), wandb.Image(xhat_valid_sagittal), wandb.Image(x_valid_sagittal), wandb.Image(diff_valid_sagittal), wandb.Image(sample_recon))
            current_table.add_data(f"Epoch {epoch} Coronal", wandb.Image(xhat_train_coronal), wandb.Image(x_train_coronal), wandb.Image(diff_train_coronal), wandb.Image(xhat_valid_coronal), wandb.Image(x_valid_coronal), wandb.Image(diff_valid_coronal), wandb.Image(sample_recon))
            wandb.log({"Epoch training/testing/diff": current_table},step = epoch + 1)
            my_table = current_table
            # --------------------------------------------------
            # ROI logging: train/val × axial/sagittal/coronal
            # --------------------------------------------------
            roi_table = wandb.Table(
                columns=[
                    "epoch",
                    "roi_id",
                    "roi_name",
                    "dataset",
                    "axis",
                    "x_slice",
                    "xhat_slice",
                    "diff_slice",
                    "image_name",
                ]
            )
            planes = {
                "axial": (
                    x_train,
                    xhat_train,
                    x_val,
                    xhat_val,
                    name_tr,
                    name_va,
                ),
                "sagittal": (
                    x_train,
                    xhat_train,
                    x_val,
                    xhat_val,
                    name_tr,
                    name_va,
                ),
                "coronal": (
                    x_train,
                    xhat_train,
                    x_val,
                    xhat_val,
                    name_tr,
                    name_va,
                ),
            }

            for axis, (x_tr, xhat_tr, x_va, xhat_va, name_tr, name_va) in planes.items():
                print(f"x_tr shape: {x_tr.shape}")
                print(f"xhat_tr shape: {xhat_tr.shape}")
                # ---------- TRAIN ----------
                log_roi_slices_to_wandb(
                    x=x_tr,
                    x_hat=xhat_tr,
                    cf=cf,
                    epoch=epoch,
                    dataset="train",
                    roi_table=roi_table,
                    label_names=label_names,
                    axis=axis,
                    max_rois=18,
                    img_name = name_tr
                )

                # ---------- VALID ----------
                log_roi_slices_to_wandb(
                    x=x_va,
                    x_hat=xhat_va,
                    cf=cf,
                    epoch=epoch,
                    dataset="val",
                    roi_table=roi_table,
                    label_names=label_names,
                    axis=axis,
                    max_rois=18,
                    img_name = name_va
                )

            wandb.log({"ROI_slices": roi_table}, step = epoch + 1)
        print(f"Total trainable parameters: {total_params_trainable}")
        cur_loss = val_total_loss if n_val_images > 0 else (train_recon_loss_avg + train_kld_loss_avg * cf.get('kld_weight', 0))
        if stop_criterion is not None:
            stop_criterion.add(cur_loss)
            if stop_criterion.stop():
                print('\nEarly Stopping!')
                break

        if cf['ckpt_enabled']:
            best_loss = save_model(model, epoch + 1, ckpt_dir, cf['save_interval'], save_best=cf['save_best'], loss=cur_loss, best_loss=best_loss)

    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'final_weights.pth'))