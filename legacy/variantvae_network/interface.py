import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
import numpy as np
from dataset.dataset_generator import ProjectionDataset
from dataset.pre_processing import compose_transforms
from utils.helper import save_as_itk
from network.models import VAE, CVAE, VQVAE, HybridVAEUNet, GMVAE2_3D, VAE_CBAM
from network.training import train_model, NormalizeImage, UnNormalizeImage, apply_label_mask
from network.losses import calc_recon_loss, calc_kld, gmvae_kld_loss
import wandb
import re
import pandas as pd
import SimpleITK as sitk

def get_class_weights(dataset, num_classes=10, epsilon=1e-6):
    """
    Compute class weights for sampling based on class frequency.
    Args:
        dataset: ProjectionDataset instance.
        num_classes: Number of classes (default: 10 for cardiac CT).
        epsilon: Small value to avoid division by zero.
    Returns:
        List of image weights for WeightedRandomSampler.
    """
    class_counts = torch.zeros(num_classes)
    image_weights = []
    for i in range(len(dataset)):
        img = dataset[i]['proj_image']
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        labels = img.view(-1).long()
        counts = torch.bincount(labels, minlength=num_classes).float()
        class_counts += counts
        weight = 1.0 / (counts[labels].sum() + epsilon)
        image_weights.append(weight.mean().item())
    proportions = (class_counts / class_counts.sum()).tolist()
    print(f"Class voxel counts: {class_counts.tolist()}")
    print(f"Class proportions: {proportions}")
    wandb.log({
        "Class Proportions": wandb.Table(
            columns=["Class", "Proportion"],
            data=[[i, p] for i, p in enumerate(proportions)]
        )
    })
    return image_weights

def extract_patient_id_from_label_path(path: str) -> str:
    fname = os.path.basename(path)
    m = re.match(r"labels_patient_(.+)\.nii\.gz$", fname)
    if m is None:
        raise ValueError(f"Could not parse patient id from {fname}")
    return m.group(1)


def get_ct_path_from_label_path(label_path: str) -> str:
    """
    Convert:
      .../labels/labels_patient_{id}.nii.gz
    to:
      .../ct/ct_patient_{id}.nii.gz
    """
    label_path = os.path.normpath(label_path)
    label_dir = os.path.dirname(label_path)
    parent_dir = os.path.dirname(label_dir)

    fname = os.path.basename(label_path)
    if not fname.startswith("labels_patient_"):
        raise ValueError(f"Unexpected label filename: {fname}")

    ct_fname = fname.replace("labels_patient_", "ct_patient_", 1)
    ct_dir = os.path.join(parent_dir, "ct")
    ct_path = os.path.join(ct_dir, ct_fname)

    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"CT file not found for {label_path}: {ct_path}")

    return ct_path


def load_ct_from_label_path(label_path: str) -> np.ndarray:
    ct_path = get_ct_path_from_label_path(label_path)
    ct_img = sitk.ReadImage(ct_path)
    return sitk.GetArrayFromImage(ct_img)


def dice_score_binary_numpy(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)

    inter = float((pred * target).sum())
    denom = float(pred.sum() + target.sum())
    return (2.0 * inter + eps) / (denom + eps)


def vol_diff_abs_numpy(a: np.ndarray, b: np.ndarray) -> int:
    return int(abs(int((a > 0).sum()) - int((b > 0).sum())))


def save_failure_case_bundle(
    label_path: str,
    input_img: np.ndarray,
    gt_img: np.ndarray,
    recon_img: np.ndarray,
    out_dir: str,
    spacing,
    direction,
):
    """
    Saves:
      - CT
      - input binary label
      - GT binary label
      - recon binary label
    """
    os.makedirs(out_dir, exist_ok=True)

    patient_id = extract_patient_id_from_label_path(label_path)
    base = f"patient_{patient_id}"

    ct_np = load_ct_from_label_path(label_path).astype(np.int16)
    input_np = input_img.astype(np.int16)
    gt_np = gt_img.astype(np.int16)
    recon_np = recon_img.astype(np.int16)

    save_as_itk(
        ct_np,
        os.path.join(out_dir, f"{base}_ct.nii.gz"),
        spacing=spacing,
        direction=direction,
    )
    save_as_itk(
        input_np,
        os.path.join(out_dir, f"{base}_input_label.nii.gz"),
        spacing=spacing,
        direction=direction,
    )
    save_as_itk(
        gt_np,
        os.path.join(out_dir, f"{base}_gt_label.nii.gz"),
        spacing=spacing,
        direction=direction,
    )
    save_as_itk(
        recon_np,
        os.path.join(out_dir, f"{base}_recon_label.nii.gz"),
        spacing=spacing,
        direction=direction,
    )

def pred_to_numpy_label_map(x_hat_i, cf, threshold=0.5):
    """
    x_hat_i: numpy array [C,D,H,W] or [1,D,H,W]
    returns: numpy int16 label map [D,H,W]
    """
    if x_hat_i.ndim != 4:
        raise ValueError(f"Expected [C,D,H,W], got {x_hat_i.shape}")

    c = x_hat_i.shape[0]

    if c == 1:
        prob = 1.0 / (1.0 + np.exp(-x_hat_i[0]))   # logits -> prob
        return (prob > threshold).astype(np.int16)

    return np.argmax(x_hat_i, axis=0).astype(np.int16)

def model_train(cf):
    """
    Train the specified VAE model using the provided configuration.
    Args:
        cf: Configuration dictionary.
    """
    print(f"Config ori_size: {cf.get('ori_size', 'Not found')}")
    if 'ori_size' not in cf:
        cf['ori_size'] = cf['proj_size'][::-1]
        print(f"Set ori_size to {cf['ori_size']} (from proj_size reversed)")

    pre_processing = compose_transforms(cf)
    # Note: return_labels=True is only for CVAE, which requires conditioning labels.
    # Current ProjectionDataset does not support returning labels; extend dataset_generator.py if needed.
    return_labels = cf['model'] == 'cvae'
    train_dataset = ProjectionDataset(
        cf['train_dir'],
        transform=pre_processing,
        sample_ratio=cf['train_sample_ratio'],
        cf = cf,
    )
    # train_weights = np.ones(len(train_dataset))
    # train_sampler = WeightedRandomSampler(train_weights, replacement=False)
    print(f"length of train dataset is {len(train_dataset)}")
    max_num_workers = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    train_loader = DataLoader(
        train_dataset,
        batch_size=cf['batch_size'],
        # sampler=train_sampler,
        num_workers=max_num_workers,
        pin_memory=True,
        shuffle=True,
    )

    if cf.get('valid', False):
        val_cf = cf.copy()
        val_cf['train'] = False
        val_pre_processing = compose_transforms(val_cf)
        val_dataset = ProjectionDataset(
            cf['valid_dir'],
            transform=val_pre_processing,
            cf = val_cf,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cf['batch_size'],
            shuffle=True,
            num_workers=max_num_workers,
            pin_memory=True
        )
    else:
        val_loader = None

    batch_sizes = [len(x['proj_image']) for x in train_loader]
    print(f"train_loader batch sizes: {batch_sizes}, total images: {sum(batch_sizes)}")

    if cf['model'] == 'paired_vae':
        assert cf['out_ch'] == 2, 'PairedVAE requires 2 output channels'
        assert cf['recon_loss'] == 'gaussian_nll', 'PairedVAE requires gaussian_nll loss'
    elif cf['model'] == 'paired_vae_label':
        assert cf['recon_loss'] == 'gaussian_nll', 'PairedVAE_label requires gaussian_nll loss'

    train_model(train_loader, val_loader, cf)
    print('\n > Training finished.')

def model_test(cf):
    """
    Test the trained model on the test dataset.
    Compatible with Post-DAE datasets that return:
      - proj_image   : model input
      - target_image : clean reconstruction target
      - proj_name    : filename
    """
    print(f"Config ori_size: {cf.get('ori_size', 'Not found')}")
    if 'ori_size' not in cf:
        cf['ori_size'] = cf['proj_size'][::-1]
        print(f"Set ori_size to {cf['ori_size']} (from proj_size reversed)")

    device = cf['device']
    if cf['model'] in ['beta_vae', 'vae', 'infovae', 'paired_vae', 'paired_vae_label', 'ae', 'soft_intro_vae']:
        model = VAE(cf, embedding=False).to(device)
    elif cf['model'] == 'cvae':
        model = CVAE(cf, embedding=False).to(device)
    elif cf['model'] == 'vqvae':
        model = VQVAE(cf, embedding=False).to(device)
    elif cf['model'] == 'hybrid_vae_unet':
        model = HybridVAEUNet(cf, embedding=False).to(device)
    elif cf['model'] == 'GMVAE2_3D':
        model = GMVAE2_3D(cf, embedding=False).to(device)
    elif cf['model'] == 'vae_cbam':
        model = VAE_CBAM(cf, embedding=False).to(device)
    else:
        raise ValueError(f"Unsupported model: {cf['model']}")

    load_weights_path = os.path.join(cf['output_folder'], 'checkpoints', cf['ckpt_file_load'])
    print(f"\n > Loading weights: {load_weights_path}")
    model.load_state_dict(torch.load(load_weights_path, map_location=device, weights_only=True))
    model.eval()

    # IMPORTANT: disable train-only Post-DAE corruption for test/train-result export
    test_cf = cf.copy()
    test_cf['train'] = False

    pre_processing = compose_transforms(test_cf)

    test_dataset = ProjectionDataset(
        cf['test_dir'],
        transform=pre_processing,
        cf=test_cf,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cf['batch_size'],
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    train_dataset = ProjectionDataset(
        cf['train_dir'],
        transform=pre_processing,
        sample_ratio=cf['train_sample_ratio'],
        cf=test_cf,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cf['batch_size'],
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    test_save_path = os.path.join(cf['output_folder'], 'test_results')
    os.makedirs(test_save_path, exist_ok=True)

    train_save_path = os.path.join(cf['output_folder'], 'train_results')
    os.makedirs(train_save_path, exist_ok=True)

    test_recon_loss, test_kld_loss = 0.0, 0.0
    input_image_slices = []
    recon_image_slices = []
    n_batches = len(test_loader)

    print(f"proj_spacing: {cf['proj_spacing']}")
    failure_rows = []
    worst_case_records = []
    num_failure_cases_to_save = cf.get('num_failure_cases_to_save', 10)

    # ----------------------------
    # Test loop
    # ----------------------------
    for batch_idx, test_batch in enumerate(test_loader):
        x_in = test_batch['proj_image'].to(device)
        x_target = test_batch.get('target_image', test_batch['proj_image']).to(device)

        x_orig = x_target.clone()
        x_orig = apply_label_mask(x_orig, cf)

        x_in = NormalizeImage(x_in, cf)

        # If Post-DAE is active, x_in is already binary {0,1}
        if cf['pre_processing'].get('post_dae_train', False) and cf.get('train', False):
            x_in = (x_in > 0).float()
        else:
            x_in = apply_label_mask(x_in, cf)

        x_target = NormalizeImage(x_target, cf)
        x_target = apply_label_mask(x_target, cf)

        with torch.no_grad():
            if cf['model'] == 'cvae':
                b_size = x_in.shape[0]
                labels = x_in.squeeze(1).long().view(b_size, -1)  # [batch_size, D*H*W]
                cond = torch.zeros(b_size, cf['out_ch'], device=device)
                for i in range(b_size):
                    counts = torch.bincount(labels[i], minlength=cf['out_ch']).float()
                    cond[i] = counts / counts.sum()
                cond = cond.to(device)
                x_hat, _, mu, log_var = model(x_in, cond)
                kld_loss = calc_kld(mu, log_var, reduction='mean', beta=cf.get('beta', 1.0)).item()

            elif cf['model'] == 'vqvae':
                x_hat, _, mu, log_var, _, _ = model(x_in)
                kld_loss = 0.0  # VQ-VAE doesn't use KL-divergence

            elif cf['model'] == 'hybrid_vae_unet':
                _, x_hat, _, mu, log_var = model(x_in)
                kld_loss = calc_kld(mu, log_var, reduction='mean', beta=cf.get('beta', 1.0)).item()

            elif cf['model'] == 'ae':
                x_hat, _, _, _ = model(x_in, deterministic=True)
                kld_loss = 0.0

            elif cf['model'] == 'GMVAE2_3D':
                x_hat, _, mu, log_var, logits_y, qy, y_idx = model(x_in)
                zm_prior = model.zm_prior
                zv_prior = model.zv_prior
                kld_loss = gmvae_kld_loss(mu, log_var, qy, zm_prior, zv_prior)

            else:
                x_hat, _, mu, log_var = model(x_in)
                kld_loss = calc_kld(mu, log_var, reduction='mean', beta=cf.get('beta', 1.0)).item()

            recon_loss = calc_recon_loss(
                x_hat,
                x_target,
                loss_type=cf['recon_loss'],
                reduction='batch_mean',
                weight_mode=cf.get('weight_mode', 'uniform'),
                alpha=cf.get('alpha', 0.5),
                epoch=cf['num_epochs']
            )
            print(f"recon_loss is {recon_loss}")
            recon_loss = recon_loss.item() if torch.is_tensor(recon_loss) else recon_loss

            test_recon_loss += recon_loss
            test_kld_loss += kld_loss

        if cf.get('save_volume', True):
            input_image_slices.append(x_in.cpu())
            recon_image_slices.append(x_hat.cpu())

        # Save each image in the batch individually
        if cf.get('save_volume', True):
            proj_names = test_batch.get('proj_name', [])
            for i in range(x_in.shape[0]):
                input_img = x_in[i].cpu().numpy()   # [1, D, H, W]
                recon_img = x_hat[i].cpu().numpy()  # [C, D, H, W]
                gt_img = x_target[i].cpu().numpy()  # [1, D, H, W]

                if cf['train_label']:
                    recon_img = pred_to_numpy_label_map(recon_img, cf, threshold=0.5)
                    input_img = (input_img.squeeze(0) > 0).astype(np.int16)
                    gt_img = (gt_img.squeeze(0) > 0).astype(np.int16)
                else:
                    input_img = UnNormalizeImage(input_img, cf)
                    input_img = input_img.squeeze(0).astype(np.int16)

                    gt_img = UnNormalizeImage(gt_img, cf)
                    gt_img = gt_img.squeeze(0).astype(np.int16)

                    recon_img = UnNormalizeImage(recon_img, cf)
                    if cf['recon_loss'] == 'gaussian_nll':
                        recon_img = recon_img[0, :, :, :]
                    else:
                        recon_img = recon_img.squeeze(0).astype(np.int16)

                exp_name = cf['exp_name']
                if proj_names and i < len(proj_names):
                    base_filename = os.path.splitext(os.path.splitext(os.path.basename(proj_names[i]))[0])[0]
                    input_filename = f'input_{base_filename}_{exp_name}.nii.gz'
                    recon_filename = f'recon_{base_filename}_{exp_name}.nii.gz'
                    gt_filename = f'gt_{base_filename}_{exp_name}.nii.gz'
                else:
                    input_filename = f'input_volume_{batch_idx}_{i}_{exp_name}.nii.gz'
                    recon_filename = f'recon_volume_{batch_idx}_{i}_{exp_name}.nii.gz'
                    gt_filename = f'gt_volume_{batch_idx}_{i}_{exp_name}.nii.gz'

                save_as_itk(
                    input_img,
                    os.path.join(test_save_path, input_filename),
                    spacing=cf['proj_spacing'],
                    direction=cf['proj_direction']
                )
                save_as_itk(
                    recon_img,
                    os.path.join(test_save_path, recon_filename),
                    spacing=cf['proj_spacing'],
                    direction=cf['proj_direction']
                )
                save_as_itk(
                    gt_img,
                    os.path.join(test_save_path, gt_filename),
                    spacing=cf['proj_spacing'],
                    direction=cf['proj_direction']
                )

                # Per-case failure metrics
                case_dice = dice_score_binary_numpy(recon_img, gt_img)
                case_input_dice = dice_score_binary_numpy(input_img, gt_img)
                case_vol_diff = vol_diff_abs_numpy(recon_img, gt_img)

                case_loss = calc_recon_loss(
                    x_hat[i:i+1],
                    x_target[i:i+1],
                    loss_type=cf['recon_loss'],
                    reduction='batch_mean',
                    weight_mode=cf.get('weight_mode', 'uniform'),
                    alpha=cf.get('alpha', 0.5),
                    epoch=cf['num_epochs']
                )
                case_loss = float(case_loss.item()) if torch.is_tensor(case_loss) else float(case_loss)

                label_path = proj_names[i] if proj_names and i < len(proj_names) else ""

                failure_rows.append({
                    "split": "test",
                    "batch_idx": batch_idx,
                    "item_idx_in_batch": i,
                    "file": os.path.basename(label_path) if label_path else recon_filename,
                    "label_path": label_path,
                    "patient_id": extract_patient_id_from_label_path(label_path) if label_path else "",
                    "dice_recon_vs_gt": case_dice,
                    "dice_input_vs_gt": case_input_dice,
                    "recon_loss": case_loss,
                    "vol_diff_abs_voxels": case_vol_diff,
                    "gt_voxels": int((gt_img > 0).sum()),
                    "recon_voxels": int((recon_img > 0).sum()),
                    "input_voxels": int((input_img > 0).sum()),
                })

                worst_case_records.append({
                    "label_path": label_path,
                    "input_img": input_img.copy(),
                    "gt_img": gt_img.copy(),
                    "recon_img": recon_img.copy(),
                    "dice_recon_vs_gt": case_dice,
                    "recon_loss": case_loss,
                })

    test_recon_loss /= max(n_batches, 1)
    test_kld_loss /= max(n_batches, 1)

    print(f"\n > Test Recon Loss: {test_recon_loss:.6f}")
    print(f" > Test KLD Loss: {test_kld_loss:.6f}")

    wandb.log({
        "Test Recon Loss": test_recon_loss,
        "Test KLD Loss": test_kld_loss,
    })
    # -----------------------------------
    # Failure-case summary + worst cases
    # -----------------------------------
    if len(failure_rows) > 0:
        failure_df = pd.DataFrame(failure_rows)

        # Worst first: lowest Dice, then highest loss
        failure_df = failure_df.sort_values(
            by=["dice_recon_vs_gt", "recon_loss", "vol_diff_abs_voxels"],
            ascending=[True, False, False]
        ).reset_index(drop=True)

        failure_csv = os.path.join(cf['output_folder'], 'test_failure_cases.csv')
        failure_df.to_csv(failure_csv, index=False)
        print(f"\n > Saved failure-case table: {failure_csv}")

        worst_dir = os.path.join(cf['output_folder'], 'worst_test_cases')
        os.makedirs(worst_dir, exist_ok=True)

        # sort records the same way
        worst_case_records = sorted(
            worst_case_records,
            key=lambda d: (d["dice_recon_vs_gt"], -d["recon_loss"])
        )

        n_save = min(num_failure_cases_to_save, len(worst_case_records))
        for rank in range(n_save):
            rec = worst_case_records[rank]
            case_dir = os.path.join(
                worst_dir,
                f"rank_{rank+1:02d}_dice_{rec['dice_recon_vs_gt']:.4f}"
            )
            save_failure_case_bundle(
                label_path=rec["label_path"],
                input_img=rec["input_img"],
                gt_img=rec["gt_img"],
                recon_img=rec["recon_img"],
                out_dir=case_dir,
                spacing=cf['proj_spacing'],
                direction=cf['proj_direction'],
            )

        print(f" > Saved worst {n_save} test cases to: {worst_dir}")
    # ----------------------------
    # Training image saving loop
    # ----------------------------
    for batch_idx, train_batch in enumerate(train_loader):
        x_in = train_batch['proj_image'].to(device)
        x_target = train_batch.get('target_image', train_batch['proj_image']).to(device)

        x_orig = x_target.clone()
        x_orig = apply_label_mask(x_orig, cf)

        x_in = NormalizeImage(x_in, cf)

        # If Post-DAE is active, x_in is already binary {0,1}
        if cf['pre_processing'].get('post_dae_train', False) and cf.get('train', False):
            x_in = (x_in > 0).float()
        else:
            x_in = apply_label_mask(x_in, cf)

        x_target = NormalizeImage(x_target, cf)
        x_target = apply_label_mask(x_target, cf)

        with torch.no_grad():
            if cf['model'] == 'cvae':
                b_size = x_in.shape[0]
                labels = x_in.squeeze(1).long().view(b_size, -1)
                cond = torch.zeros(b_size, cf['out_ch'], device=device)
                for i in range(b_size):
                    counts = torch.bincount(labels[i], minlength=cf['out_ch']).float()
                    cond[i] = counts / counts.sum()
                cond = cond.to(device)
                x_hat, _, mu, log_var = model(x_in, cond)

            elif cf['model'] == 'vqvae':
                x_hat, _, mu, log_var, _, _ = model(x_in)

            elif cf['model'] == 'hybrid_vae_unet':
                _, x_hat, _, mu, log_var = model(x_in)

            elif cf['model'] == 'ae':
                x_hat, _, _, _ = model(x_in, deterministic=True)

            elif cf['model'] == 'GMVAE2_3D':
                x_hat, _, mu, log_var, logits_y, qy, y_idx = model(x_in)

            else:
                x_hat, _, mu, log_var = model(x_in)

            recon_loss = calc_recon_loss(
                x_hat,
                x_target,
                loss_type=cf['recon_loss'],
                reduction='batch_mean',
                weight_mode=cf.get('weight_mode', 'uniform'),
                alpha=cf.get('alpha', 0.5),
                epoch=cf['num_epochs']
            )
            print(f"train recon_loss is {recon_loss}")
            recon_loss = recon_loss.item() if torch.is_tensor(recon_loss) else recon_loss

        if cf.get('save_volume', True):
            proj_names = train_batch.get('proj_name', [])
            for i in range(x_in.shape[0]):
                input_img = x_target[i].cpu().numpy()
                recon_img = x_hat[i].cpu().numpy()

                if cf['train_label']:
                    recon_img = pred_to_numpy_label_map(recon_img, cf, threshold=0.5)
                    input_img = (input_img.squeeze(0) > 0).astype(np.int16)
                else:
                    input_img = UnNormalizeImage(input_img, cf)
                    input_img = input_img.squeeze(0).astype(np.int16)

                    input_img_orig = x_orig[i].cpu().numpy()
                    input_img_orig = input_img_orig.squeeze(0).astype(np.int16)
                    if not np.allclose(input_img, input_img_orig):
                        print(f"Train input does not match x_orig for batch {batch_idx}, image {i}")
                        print(f"input_img slice: {input_img[50, 50, 0:50]}")
                        print(f"input_img_orig slice: {input_img_orig[50, 50, 0:50]}")

                    recon_img = UnNormalizeImage(recon_img, cf)
                    if cf['recon_loss'] == 'gaussian_nll':
                        recon_img = recon_img[0, :, :, :]
                    else:
                        recon_img = recon_img.squeeze(0).astype(np.int16)

                exp_name = cf['exp_name']
                if proj_names and i < len(proj_names):
                    base_filename = os.path.splitext(os.path.splitext(os.path.basename(proj_names[i]))[0])[0]
                    input_filename = f'input_{base_filename}_{exp_name}.nii.gz'
                    recon_filename = f'recon_{base_filename}_{exp_name}.nii.gz'
                else:
                    input_filename = f'input_volume_{batch_idx}_{i}_{exp_name}.nii.gz'
                    recon_filename = f'recon_volume_{batch_idx}_{i}_{exp_name}.nii.gz'

                save_as_itk(
                    input_img,
                    os.path.join(train_save_path, input_filename),
                    spacing=cf['proj_spacing'],
                    direction=cf['proj_direction']
                )
                save_as_itk(
                    recon_img,
                    os.path.join(train_save_path, recon_filename),
                    spacing=cf['proj_spacing'],
                    direction=cf['proj_direction']
                )

    print("\n > Testing finished.")