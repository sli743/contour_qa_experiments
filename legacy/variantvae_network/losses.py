import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import math

def calc_mae(x, y, reduction='mean'):
    if reduction == 'none':
        return F.l1_loss(x, y, reduction='none')
    elif reduction == 'mean':
        return F.l1_loss(x, y, reduction='mean')
    elif reduction == 'batch_mean':
        batch_size = x.shape[0]
        return F.l1_loss(x, y, reduction='sum') / batch_size
    elif reduction == 'sum':
        return F.l1_loss(x, y, reduction='sum')
    else:
        raise NotImplementedError

def calc_mse(x, y, reduction='mean'):
    if reduction == 'none':
        return F.mse_loss(x, y, reduction='none')
    elif reduction == 'mean':
        return F.mse_loss(x, y, reduction='mean')
    elif reduction == 'batch_mean':
        batch_size = x.shape[0]
        return F.mse_loss(x, y, reduction='sum') / batch_size
    elif reduction == 'sum':
        return F.mse_loss(x, y, reduction='sum')
    else:
        raise NotImplementedError

def calc_bce(x, y, reduction='mean'):
    if reduction == 'none':
        return F.binary_cross_entropy_with_logits(x, y, reduction='none')
    elif reduction == 'mean':
        return F.binary_cross_entropy_with_logits(x, y, reduction='mean')
    elif reduction == 'batch_mean':
        batch_size = x.shape[0]
        return F.binary_cross_entropy_with_logits(x, y, reduction='sum') / batch_size
    elif reduction == 'sum':
        return F.binary_cross_entropy_with_logits(x, y, reduction='sum')
    else:
        raise NotImplementedError

def calc_ce(x, y, reduction='mean'):
    if reduction == 'none':
        return F.cross_entropy(x, y, reduction='none')
    elif reduction == 'mean':
        return F.cross_entropy(x, y, reduction='mean')
    elif reduction == 'batch_mean':
        batch_size = x.shape[0]
        return F.cross_entropy(x, y, reduction='sum') / batch_size
    elif reduction == 'sum':
        return F.cross_entropy(x, y, reduction='sum')
    else:
        raise NotImplementedError

def calc_nll(x, y, reduction='mean'):
    var = torch.mean((y - x) ** 2)
    if reduction == 'none':
        return F.gaussian_nll_loss(x, y, var, reduction='none')
    elif reduction == 'mean':
        return F.gaussian_nll_loss(x, y, var, reduction='mean')
    elif reduction == 'batch_mean':
        batch_size = x.shape[0]
        return F.gaussian_nll_loss(x, y, var, reduction='sum') / batch_size
    elif reduction == 'sum':
        return F.gaussian_nll_loss(x, y, var, reduction='sum')
    else:
        raise NotImplementedError

def calc_gaussian_nll(mu, y, var, reduction='mean'):
    if reduction == 'none':
        return F.gaussian_nll_loss(mu, y, var, reduction='none')
    elif reduction == 'mean':
        return F.gaussian_nll_loss(mu, y, var, reduction='mean')
    elif reduction == 'batch_mean':
        batch_size = mu.shape[0]
        return F.gaussian_nll_loss(mu, y, var, reduction='sum') / batch_size
    elif reduction == 'sum':
        return F.gaussian_nll_loss(mu, y, var, reduction='sum')
    else:
        raise NotImplementedError

def compute_label_mask(x, N):
    """
    x: [B, 1, X, Y, Z] with 0-N labels
    Returns:
        mask_bn: [B, N] float tensor, 1 = non-empty label, 0 = empty
    """
    if x.shape[1] == 1:
        x = F.one_hot(x.squeeze(1).long(), num_classes=N).permute(0, 4, 1, 2, 3).float()
    return (x.sum(dim=(2, 3, 4)) > 0).float()

def binary_soft_dice_loss_from_logits(logits, target, eps=1e-6):
    """
    logits: [B, 1, D, H, W]
    target: [B, 1, D, H, W] with {0,1}
    returns: [B]
    """
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum(dim=(1, 2, 3, 4))
    denominator = prob.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4)) + eps
    dice_loss = 1.0 - (2.0 * intersection + eps) / denominator
    return dice_loss

def calc_recon_loss(x_hat, x, loss_type='mse', reduction='mean', weight_mode='uniform', return_per_class=False, alpha=0.5, gamma=2.0, warmup_epochs=50, schedule="linear", epoch=0):
    if x_hat.shape[0] != x.shape[0] or x_hat.shape[2:] != x.shape[2:]:
        raise ValueError(f"Images should have same shape, but got {x_hat.shape} and {x.shape}")
    
    if loss_type == 'dice_bce':
        if x_hat.shape[1] != 1:
            raise ValueError(f"dice_bce expects x_hat with 1 channel, got {x_hat.shape}")

        target = (x > 0).float()
        dice_loss = binary_soft_dice_loss_from_logits(x_hat, target)   # [B]
        bce_loss = calc_bce(x_hat, target, reduction='none').mean(dim=(1, 2, 3, 4))  # [B]
        print(f"dice_loss: {dice_loss}, bce_loss: {bce_loss}")
        loss = alpha * dice_loss + (1 - alpha) * bce_loss   # [B]

        if return_per_class:
            loss = loss.unsqueeze(1)   # [B, 1]
            if reduction == 'none':
                return loss
            elif reduction == 'mean':
                return loss
            elif reduction == 'batch_mean':
                return loss.mean()
            elif reduction == 'sum':
                return loss.sum()
            else:
                raise NotImplementedError
        else:
            if reduction == 'none':
                return loss
            elif reduction in ['mean', 'batch_mean']:
                return loss.mean()
            elif reduction == 'sum':
                return loss.sum()
            else:
                raise NotImplementedError

    # --------------------------------------------------
    # Mask for empty label volumes (B, N)
    # --------------------------------------------------
    N = x_hat.shape[1]
    label_mask = compute_label_mask(x, N)
    # if label_mask[k,:] is not all ones, the set label_mask[k,0] as 0 for background to avoid biased learning
    for k in range(x.shape[0]):
        if not torch.all(label_mask[k,:] == 1):
            label_mask[k,0] = 0
    # shape: [B, N]
    if loss_type in ['dice', 'dice_entropy', 'dice_focal']:
        channel_sum = x_hat.sum(dim=1)

        if not torch.allclose(channel_sum, torch.ones_like(x_hat[:, 0]), rtol=1e-3, atol=1e-3):
            print(f"Invalid x_hat sum across channels: min {channel_sum.min().item()}, max {channel_sum.max().item()}")
            x_hat = F.softmax(x_hat, dim=1)

    if loss_type in ['mae', 'mse', 'bce', 'nll', 'gaussian_nll']:
        # Always compute unreduced voxel loss
        if loss_type == 'mae':
            voxel_loss = calc_mae(x_hat, x, reduction='none')
        elif loss_type == 'mse':
            voxel_loss = calc_mse(x_hat, x, reduction='none')
        elif loss_type == 'bce':
            voxel_loss = calc_bce(x_hat, x, reduction='none')
        elif loss_type == 'nll':
            voxel_loss = calc_nll(x_hat, x, reduction='none')
        elif loss_type == 'gaussian_nll':
            assert x_hat.shape[1] == 2
            mu_x = x_hat[:, [0], ...]
            log_var = x_hat[:, [1], ...]
            var = torch.exp(log_var)
            voxel_loss = calc_gaussian_nll(mu_x, x, var, reduction='none')

        # Reduce spatial dims → [B, N]
        loss_bn = voxel_loss.mean(dim=(2, 3, 4))

        # Apply label mask
        loss_bn = loss_bn * label_mask
        print(f"loss_bn is {loss_bn}")

        # Final reduction
        if reduction == 'none':
            res = loss_bn
        elif reduction in ['mean', 'batch_mean']:
            res = loss_bn.sum() / (label_mask.sum() + 1e-8)
        elif reduction == 'sum':
            res = loss_bn.sum()
        print(f"res is {res}")

    elif loss_type in ['dice', 'dice_entropy', 'dice_focal', 'binary_soft_dice_from_logits']:
        if x.shape[1] == 1:
            x = F.one_hot(x.squeeze(1).long(), num_classes=x_hat.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        if weight_mode == 'inverse_frequency':
            class_counts = x.sum(dim=(0, 2, 3, 4)) + 1e-6
            weights = 1.0 / class_counts
            weights = weights / weights.sum() * x.shape[1]
            weights = torch.clamp(weights, max=2.0)
        elif weight_mode == 'sqrt_inverse_frequency':
            class_counts = x.sum(dim=(0, 2, 3, 4)) + 1e-6
            weights = 1.0 / class_counts.sqrt()
            weights = weights / weights.sum() * x.shape[1]
            weights = torch.clamp(weights, max=2.0)
        else:
            weights = torch.ones(x.shape[1], device=x.device)

        intersection = (x_hat * x).sum(dim=(2, 3, 4))
        denominator = (x_hat.sum(dim=(2, 3, 4)) + x.sum(dim=(2, 3, 4))) + 1e-6
        dice = 1 - (2 * intersection + 1e-6) / denominator
        # dice shape: [B, N]
        dice = dice * label_mask


        if loss_type == 'dice':
            loss = dice
        elif loss_type == 'dice_entropy':
            ce_loss = -x * torch.log(x_hat.clamp(min=1e-6))
            ce_loss = ce_loss.mean(dim=(2, 3, 4))
            loss = alpha * dice + (1 - alpha) * ce_loss
        elif loss_type == 'dice_focal':
            ce_loss = -x * torch.log(x_hat.clamp(min=1e-6))
            focal_loss = (1 - x_hat) ** gamma * ce_loss
            focal_loss = focal_loss.sum(dim=(2, 3, 4)) / (x.sum(dim=(2, 3, 4)) + 1e-6)
            loss = alpha * dice + (1 - alpha) * focal_loss

        weighted_loss = loss * weights

        if return_per_class:
            if reduction == 'none':
                return weighted_loss
            elif reduction == 'mean':
                return weighted_loss
            elif reduction == 'batch_mean':
                return weighted_loss.mean()
        else:
            loss = weighted_loss.sum(dim=1) / (label_mask.sum(dim=1) + 1e-8)
            if reduction == 'none':
                return loss
            elif reduction in ['mean', 'batch_mean']:
                return loss.mean()

    elif loss_type == 'hausdorff':
        if x.shape[1] == 1:
            x = F.one_hot(x.squeeze(1).long(), num_classes=x_hat.shape[1]).permute(0, 4, 1, 2, 3).float()
        criterion = HybridDiceHausdorffLoss(warmup_epochs=warmup_epochs, schedule=schedule, reduction=reduction, return_per_class=return_per_class)
        res = criterion(x_hat, x, epoch)
    elif loss_type == 'hausdorff_95':
        if x.shape[1] == 1:
            x = F.one_hot(x.squeeze(1).long(), num_classes=x_hat.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        # Check if x and x_hat are one-hot vectors
        # print(f"x shape: {x.shape}, x_hat shape: {x_hat.shape}")
        # print(f"x unique values: {torch.unique(x)}")
        # print(f"x_hat min: {x_hat.min().item():.4f}, max: {x_hat.max().item():.4f}")
        # print(f"x_hat unique values (first 10): {torch.unique(x_hat).flatten()[:10]}")
        
        # Check if x is one-hot (only 0s and 1s)
        x_is_onehot = torch.all((x == 0) | (x == 1))
        # print(f"x is one-hot: {x_is_onehot}")
        
        # Check if x_hat is one-hot (only 0s and 1s)
        x_hat_is_onehot = torch.all((x_hat == 0) | (x_hat == 1))
        # print(f"x_hat is one-hot: {x_hat_is_onehot}")
        
        # Check channel sums for x_hat
        channel_sum = x_hat.sum(dim=1)
        # print(f"x_hat channel sums - min: {channel_sum.min().item():.4f}, max: {channel_sum.max().item():.4f}")
        
        from surface_distance import metrics
        
        # Convert predictions to binary masks
        # Since x_hat has very small values, use argmax instead of thresholding
        # print("Using argmax to convert probabilities to binary masks")
        x_hat_binary = F.one_hot(torch.argmax(x_hat, dim=1), num_classes=x_hat.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        # Verify binary conversion
        # print(f"x_hat_binary shape: {x_hat_binary.shape}")
        # print(f"x_hat_binary unique values: {torch.unique(x_hat_binary)}")
        # print(f"x_hat_binary sums per class: {x_hat_binary.sum(dim=(2,3,4))}")
        # print(f"x shape: {x.shape}")
        # print(f"x unique values: {torch.unique(x)}")
        # print(f"x sums per class: {x.sum(dim=(2,3,4))}")
        
        batch_size, num_classes = x.shape[:2]
        hausdorff_distances = torch.zeros(num_classes, device=x.device)
        
        for class_idx in range(num_classes):
            class_distances = []
            
            for batch_idx in range(batch_size):
                pred_mask = x_hat_binary[batch_idx, class_idx].cpu().numpy().astype(bool)
                gt_mask = x[batch_idx, class_idx].cpu().numpy().astype(bool)
                
                # print(f"Class {class_idx}, Batch {batch_idx}: pred_sum={pred_mask.sum()}, gt_sum={gt_mask.sum()}")
                
                # Skip if both masks are empty
                if pred_mask.sum() == 0 and gt_mask.sum() == 0:
                    # print(f"  Both masks empty -> distance = 0.0")
                    class_distances.append(0.0)
                    continue
                # If one mask is empty, use maximum possible distance
                elif pred_mask.sum() == 0 or gt_mask.sum() == 0:
                    # print(f"  One mask empty -> distance = inf")
                    max_dist = np.inf
                    class_distances.append(max_dist)
                    continue
                
                # Compute surface distances
                try:
                    # print(f"  Computing surface distances...")
                    # Get surface points
                    pred_surface = metrics.compute_surface_distances(pred_mask, gt_mask, (1.0, 1.0, 1.0))
                    # print(f"  Surface distances computed, shape: {pred_surface.shape if hasattr(pred_surface, 'shape') else type(pred_surface)}")
                    
                    # Compute Hausdorff 95th percentile distance
                    hd95 = metrics.compute_robust_hausdorff(pred_surface, 95.0)
                    # print(f"  HD95 computed: {hd95}")
                    class_distances.append(hd95)
                except Exception as e:
                    print(f"  Exception in surface distance computation: {e}")
                    # Fallback to maximum distance if computation fails
                    max_dist = np.inf
                    class_distances.append(max_dist)
            # Average over batch for this class
            hausdorff_distances[class_idx] = torch.tensor(np.mean(class_distances), device=x.device)
        print(f'hausdorff_distances: {hausdorff_distances}')
        res = hausdorff_distances
        
        # Apply return_per_class and reduction logic
        if not return_per_class:
            if reduction == 'none':
                return res
            elif reduction == 'mean':
                return res.mean()
            elif reduction == 'batch_mean':
                return res.mean()
            else:
                return res.mean()
        else:
            return [res]
            
    elif loss_type == 'surface_dice_2mm':
        if x.shape[1] == 1:
            x = F.one_hot(x.squeeze(1).long(), num_classes=x_hat.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        from surface_distance import metrics
        import numpy as np
        
        # Convert predictions to binary masks
        # Since x_hat has very small values, use argmax instead of thresholding
        print("Using argmax to convert probabilities to binary masks (surface_dice_2mm)")
        x_hat_binary = F.one_hot(torch.argmax(x_hat, dim=1), num_classes=x_hat.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        batch_size, num_classes = x.shape[:2]
        surface_dice_distances = torch.zeros(num_classes, device=x.device)
        
        for class_idx in range(num_classes):
            class_distances = []
            
            for batch_idx in range(batch_size):
                pred_mask = x_hat_binary[batch_idx, class_idx].cpu().numpy().astype(bool)
                gt_mask = x[batch_idx, class_idx].cpu().numpy().astype(bool)
                
                # Skip if both masks are empty
                if pred_mask.sum() == 0 and gt_mask.sum() == 0:
                    class_distances.append(1.0)  # Perfect dice when both empty
                    continue
                # If one mask is empty, surface dice is 0
                elif pred_mask.sum() == 0 or gt_mask.sum() == 0:
                    class_distances.append(0.0)
                    continue
                
                # Compute surface distances
                try:
                    # Get surface points
                    distances = metrics.compute_surface_distances(gt_mask, pred_mask, (1.0, 1.0, 1.0))
                    # Compute surface dice at 2mm tolerance
                    s_dice_2mm = metrics.compute_surface_dice_at_tolerance(distances, 2.0)
                    class_distances.append(s_dice_2mm)
                except Exception as e:
                    # Fallback to 0 if computation fails
                    class_distances.append(0.0)
            
            # Average over batch for this class
            surface_dice_distances[class_idx] = torch.tensor(np.mean(class_distances), device=x.device)
        
        res = surface_dice_distances
        
        # Apply return_per_class and reduction logic
        if not return_per_class:
            if reduction == 'none':
                return res
            elif reduction == 'mean':
                return res.mean()
            elif reduction == 'batch_mean':
                return res.mean()
            else:
                return res.mean()
        else:
            return res
            
    else:
        raise NotImplementedError(f'Unknown loss type: {loss_type}')

    return res


def soft_dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2 * torch.sum(pred * target, dim=(0,2,3,4))
    den = torch.sum(pred, dim=(0,2,3,4)) + torch.sum(target, dim=(0,2,3,4)) + eps
    dice_per_class = 1 - num / den
    return dice_per_class

def distance_transform(binary_mask):
    """
    Chamfer-style distance transform (approximate).
    binary_mask: [N,C,D,H,W] with {0,1}.
    """
    dist = 1 - binary_mask
    max_val = dist.shape[2] + dist.shape[3] + dist.shape[4]
    dist = dist * max_val
    for _ in range(4):
        dist = torch.min(dist, 1 + F.avg_pool3d(dist, kernel_size=3, stride=1, padding=1))
    return dist

def surface_loss(pred, target, eps=1e-6):
    N, C, D, H, W = pred.shape
    dmax = torch.sqrt(torch.tensor(D**2 + H**2 + W**2, dtype=torch.float32, device=pred.device))

    pred_prob = torch.sigmoid(pred)
    pred_bin = (pred_prob > 0.5).float()

    dt_gt = distance_transform(target) / dmax
    dt_pred = distance_transform(pred_bin) / dmax

    fp = torch.sum(pred_prob * dt_gt, dim=(0,2,3,4)) / (torch.sum(pred_prob, dim=(0,2,3,4)) + eps)
    fn = torch.sum((1 - pred_prob) * dt_pred, dim=(0,2,3,4)) / (torch.sum(1 - pred_prob, dim=(0,2,3,4)) + eps)
    return fp + fn

class HybridDiceHausdorffLoss(nn.Module):
    def __init__(self, warmup_epochs=5, schedule="linear", reduction="mean", return_per_class=False):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.schedule = schedule
        self.reduction = reduction
        self.return_per_class = return_per_class

    def alpha(self, epoch: int) -> float:
        if self.schedule == "linear":
            return min(1.0, epoch / self.warmup_epochs)
        elif self.schedule == "cosine":
            return 0.5 * (1 - torch.cos(torch.tensor(epoch / self.warmup_epochs * 3.14159)))
        elif self.schedule == "exp":
            return 1 - torch.exp(-epoch / (self.warmup_epochs / 5))
        else:
            raise ValueError("Unknown schedule")

    def forward(self, pred, target, epoch: int):
        dice_per_class = soft_dice_loss(pred, target)
        print(f'dice_per_class: {dice_per_class}')
        surf_per_class = surface_loss(pred, target)
        print(f'surf_per_class: {surf_per_class}')

        a = self.alpha(epoch)
        loss_per_class = (1 - a) * dice_per_class + a * surf_per_class

        if self.return_per_class:
            return loss_per_class  # shape [C]

        if self.reduction in ["mean", "batch_mean"]:
            return torch.mean(loss_per_class)
        elif self.reduction == "sum":
            return torch.sum(loss_per_class)
        else:
            return loss_per_class


def gmvae_kld_loss(mu, var, qy, zm_prior, zv_prior, reduction='mean'):
    """
    KL[q(z|x) || p(z|y)] averaged over q(y|z)
    + KL[q(y|z) || p(y)] where p(y) is uniform.

    Args:
        mu:     [B, z_dim] - mean of q(z|x)
        var:    [B, z_dim] - variance of q(z|x)
        qy:     [B, K] - categorical probabilities of q(y|z)
        zm_prior: list of [1, z_dim] tensors for each k
        zv_prior: list of [1, z_dim] tensors for each k
        reduction: 'mean' | 'sum' | 'none'

    Returns:
        scalar tensor (if reduction is not 'none'), else vector [B]
    """
    B, z_dim = mu.shape
    K = qy.shape[1]
    device = mu.device

    # KL[q(z|x) || p(z|y)] for each component
    kld_z_all = torch.zeros(B, device=device)

    for k in range(K):
        mu_k = zm_prior[k].expand(B, -1)
        var_k = zv_prior[k].expand(B, -1)

        kl_k = 0.5 * torch.sum(
            torch.log(var_k + 1e-8) - torch.log(var + 1e-8) +
            (var + (mu - mu_k) ** 2) / (var_k + 1e-8) - 1,
            dim=1  # sum over latent dim → shape: [B]
        )
        kld_z_all += qy[:, k] * kl_k

    # KL[q(y|z) || p(y)] — uniform p(y)
    log_qy = torch.log(qy + 1e-8)
    kl_y = torch.sum(qy * (log_qy - math.log(1.0 / K)), dim=1)  # [B]

    # Total KL per sample
    total_kl = kld_z_all + kl_y  # [B]

    # Apply reduction
    if reduction == 'none':
        return total_kl
    elif reduction == 'sum':
        return total_kl.sum()
    elif reduction == 'mean':
        return total_kl.mean()
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def calc_kld(mu, log_var, reduction='mean', beta=1.0):
    if mu.shape != log_var.shape or len(mu.shape) != 2:
        raise ValueError(f"Expected mu, log_var: {mu.shape}, {log_var.shape}")
    kld = -0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=1)
    kld = kld * beta
    if reduction == 'mean':
        kld = torch.mean(kld)
    elif reduction == 'sum':
        kld = torch.sum(kld)
    return kld

def calc_mmd(z, z_prior, kernel='rbf', sigma=1.0):
    batch_size = z.shape[0]
    z = z.view(batch_size, -1)
    z_prior = z_prior.view(batch_size, -1)
    
    if kernel == 'rbf':
        xx = torch.matmul(z, z.t())
        xy = torch.matmul(z, z_prior.t())
        yy = torch.matmul(z_prior, z_prior.t())
        
        rx = (xx.diag().unsqueeze(0) - 2 * xx + xx.diag().unsqueeze(1))
        ry = (yy.diag().unsqueeze(0) - 2 * yy + yy.diag().sum(1))
        rxy = (xx.diag().unsqueeze(0) - 2 * xy + yy.diag().unsqueeze(1))
        
        K = torch.exp(-rx / (2 * sigma ** 2))
        L = torch.exp(-ry / (2 * sigma ** 2))
        P = torch.exp(-rxy / (2 * sigma ** 2))
        
        mmd = K.mean() + L.sum() - 2 * P.mean()
        return mmd
    else:
        raise NotImplementedError(f"Kernel {kernel} not supported")

class EmbeddingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.similarity = nn.CosineSimilarity()

    def forward(self, teacher_embeddings, student_embeddings):
        layer_id = 0
        for teacher_feature, student_feature in zip(teacher_embeddings, student_embeddings):
            if layer_id == 0:
                total_loss = 0.5 * self.criterion(teacher_feature, student_feature)
            else:
                total_loss += 0.5 * self.criterion(teacher_feature, student_feature)
            total_loss += torch.mean(1 - self.similarity_loss(teacher_feature.view(teacher_feature.shape[0], -1),
                                                             student_feature.view(student_feature.shape[0], -1)))
            layer_id += 1
        return total_loss

def compute_class_weights(targets, num_classes, mode='uniform', epsilon=1e-6):
    flat_targets = targets.flatten(-1)
    class_counts = torch.bincount(flat_targets, minlength=num_classes).float()
    if mode == 'uniform':
        return torch.ones(num_classes, device=targets.device)
    elif mode == 'inverse_frequency':
        weights = 1.0 / (class_counts + epsilon)
        return weights / weights.sum() * num_classes
    elif mode == 'sqrt_inverse_frequency':
        weights = 1.0 / (class_counts.sqrt() + epsilon)
        return weights / weights.sum() * num_classes
    else:
        raise ValueError(f"Unknown class weighting mode: {mode}")

def dice_loss_3d(preds, targets, weight_mode='uniform', custom_weights=None, smooth=1e-6, return_per_class=False):
    num_classes = preds.shape[1]
    if preds.min() < 0 or preds.max() > 1:
        preds = F.softmax(preds, dim=1)
    targets_one_hot = F.one_hot(targets.long(), num_classes=preds.shape[1]).movedim(-1).squeeze(2).float()
    targets = targets.long()
    intersection = (preds * targets_one_hot).sum(dim=(2, 3, 4))
    union = torch.sum(preds, dim=(2, 3, 4)) + torch.sum(targets_one_hot, dim=(2, 3, 4))
    dice_scores = (2. * intersection + smooth) / (union + smooth)
    dice_losses = 1 - dice_scores
    if weight_mode == 'custom':
        if custom_weights is None:
            raise ValueError("Custom weights must be provided for weight_mode='custom'")
        class_weights = custom_weights.to(preds.device)
    else:
        class_weights = compute_class_weights(targets, num_classes, mode=weight_mode)
    if return_per_class:
        return dice_losses
    else:
        weighted_loss = dice_losses * class_weights
        return weighted_loss.mean()