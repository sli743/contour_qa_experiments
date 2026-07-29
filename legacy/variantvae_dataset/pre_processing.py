import torch
import torch.nn.functional as F_nn
import numpy as np
from torchvision.transforms import v2


class MinMaxNormalize(object):
    """Normalize the input data to the given range."""
    def __init__(self, out_min=0, out_max=1, clip=False, clip_percent=0.5):
        assert out_min < out_max
        self.out_min = out_min
        self.out_max = out_max
        self.clip = clip
        self.clip_percent = clip_percent

    def __call__(self, image):
        if not self.clip:
            data_min = np.min(image)
            data_max = np.max(image)
        else:
            array = image.flatten()
            totalsize = np.size(array)
            lower_bound = int(self.clip_percent / 100 * totalsize)
            upper_bound = int((100 - self.clip_percent) / 100 * totalsize)
            data_min = array[np.argpartition(array, lower_bound)[lower_bound]]
            data_max = array[np.argpartition(array, upper_bound)[upper_bound]]
            image = np.clip(image, data_min, data_max)

        image = (image - data_min) / (data_max - data_min) * (self.out_max - self.out_min) + self.out_min
        return image


class ToDtype_3d(object):
    def __init__(self, dt):
        self.dt = dt

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        return image.to(dtype=getattr(torch, self.dt))


class NaturalNormalize(object):
    def __init__(self, MAX_PIXEL_VAL=256):
        self.MAX_PIXEL_VAL = MAX_PIXEL_VAL

    def __call__(self, image):
        return image / self.MAX_PIXEL_VAL


def _max_pool3d_same(x, kernel_size):
    pad = kernel_size // 2
    return F_nn.max_pool3d(x, kernel_size=kernel_size, stride=1, padding=pad)


def binary_dilate3d(x, kernel_size=3, iterations=1):
    """
    x: [1, D, H, W] binary tensor
    """
    x = x.unsqueeze(0)  # [1,1,D,H,W]
    for _ in range(iterations):
        x = _max_pool3d_same(x, kernel_size)
        x = (x > 0).float()
    return x.squeeze(0)


def binary_erode3d(x, kernel_size=3, iterations=1):
    """
    x: [1, D, H, W] binary tensor
    """
    x = x.unsqueeze(0)
    for _ in range(iterations):
        x_inv = 1.0 - x
        x_inv = _max_pool3d_same(x_inv, kernel_size)
        x = 1.0 - x_inv
        x = (x > 0).float()
    return x.squeeze(0)


def random_shift3d(x, max_shift=3):
    """
    x: [1, D, H, W]
    shift with zero padding
    """
    dz = np.random.randint(-max_shift, max_shift + 1)
    dy = np.random.randint(-max_shift, max_shift + 1)
    dx = np.random.randint(-max_shift, max_shift + 1)

    out = torch.zeros_like(x)
    _, D, H, W = x.shape

    z_src0 = max(0, -dz)
    z_src1 = min(D, D - dz) if dz >= 0 else D
    z_dst0 = max(0, dz)
    z_dst1 = z_dst0 + (z_src1 - z_src0)

    y_src0 = max(0, -dy)
    y_src1 = min(H, H - dy) if dy >= 0 else H
    y_dst0 = max(0, dy)
    y_dst1 = y_dst0 + (y_src1 - y_src0)

    x_src0 = max(0, -dx)
    x_src1 = min(W, W - dx) if dx >= 0 else W
    x_dst0 = max(0, dx)
    x_dst1 = x_dst0 + (x_src1 - x_src0)

    out[:, z_dst0:z_dst1, y_dst0:y_dst1, x_dst0:x_dst1] = \
        x[:, z_src0:z_src1, y_src0:y_src1, x_src0:x_src1]

    return out


def zoom_in_3d(image, scale=1.08):
    """
    Small zoom-in for [C, D, H, W].
    Keeps output size unchanged.
    Uses nearest-neighbor because labels are discrete.
    """
    if image.dim() != 4:
        raise ValueError(f"Expected [C,D,H,W], got {image.shape}")

    image_b = image.unsqueeze(0)  # [1,C,D,H,W]

    theta = torch.tensor(
        [[
            [1.0 / scale, 0.0, 0.0, 0.0],
            [0.0, 1.0 / scale, 0.0, 0.0],
            [0.0, 0.0, 1.0 / scale, 0.0],
        ]],
        dtype=image.dtype,
        device=image.device,
    )

    grid = F_nn.affine_grid(theta, size=image_b.shape, align_corners=False)
    out = F_nn.grid_sample(
        image_b,
        grid,
        mode="nearest",
        padding_mode="border",
        align_corners=False,
    )

    return out.squeeze(0)


def rotate_3d_z(image, angle_degrees):
    """
    Rotate [C, D, H, W] around the z-axis.
    Uses nearest-neighbor because labels are discrete.
    """
    if image.dim() != 4:
        raise ValueError(f"Expected [C,D,H,W], got {image.shape}")

    image_b = image.unsqueeze(0)  # [1,C,D,H,W]

    angle_rad = torch.deg2rad(torch.tensor(angle_degrees, dtype=image.dtype, device=image.device))
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    theta = torch.tensor(
        [[
            [cos_a.item(), -sin_a.item(), 0.0, 0.0],
            [sin_a.item(),  cos_a.item(), 0.0, 0.0],
            [0.0,           0.0,          1.0, 0.0],
        ]],
        dtype=image.dtype,
        device=image.device,
    )

    grid = F_nn.affine_grid(theta, size=image_b.shape, align_corners=False)
    out = F_nn.grid_sample(
        image_b,
        grid,
        mode="nearest",
        padding_mode="border",
        align_corners=False,
    )

    return out.squeeze(0)


class StandardAugmentation3D(object):
    """
    Mild standard augmentation for label masks.
    Safe for left-lung task: no left-right flip.
    Supports small zoom, small rotation, and small translation.
    """
    def __init__(
        self,
        prob=0.5,
        zoom_prob=0.3,
        zoom_min=1.03,
        zoom_max=1.10,
        rotate_prob=0.0,
        rotate_min_deg=-5.0,
        rotate_max_deg=5.0,
        translate_prob=0.0,
        translate_max_shift=3,
        require_foreground=True,
    ):
        self.prob = prob
        self.zoom_prob = zoom_prob
        self.zoom_min = zoom_min
        self.zoom_max = zoom_max
        self.rotate_prob = rotate_prob
        self.rotate_min_deg = rotate_min_deg
        self.rotate_max_deg = rotate_max_deg
        self.translate_prob = translate_prob
        self.translate_max_shift = translate_max_shift
        self.require_foreground = require_foreground

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        image = image.float()

        if self.require_foreground and not torch.any(image > 0):
            return image

        if torch.rand(1).item() >= self.prob:
            return image

        x = image.clone()

        if torch.rand(1).item() < self.zoom_prob:
            scale = np.random.uniform(self.zoom_min, self.zoom_max)
            x = zoom_in_3d(x, scale=scale)

        if torch.rand(1).item() < self.rotate_prob:
            angle = np.random.uniform(self.rotate_min_deg, self.rotate_max_deg)
            x = rotate_3d_z(x, angle_degrees=angle)

        if torch.rand(1).item() < self.translate_prob:
            x = random_shift3d(x, max_shift=self.translate_max_shift)

        return torch.round(x)


class PostDAECorruption3D(object):
    """
    Mixed clean/corrupted input for Post-DAE training.

    Input/Output shape: [1, D, H, W]
    Assumes foreground is image > 0.

    Behavior:
      - with probability clean_prob: return clean binary mask
      - otherwise, with probability prob: return corrupted binary mask
      - otherwise: return clean binary mask
    """
    def __init__(
        self,
        prob=0.5,
        clean_prob=0.5,
        shift_prob=0.3,
        erode_prob=0.2,
        dilate_prob=0.2,
        max_shift=3,
        morph_kernel=3,
        morph_iters=(1, 1),
        require_foreground=True,
    ):
        self.prob = prob
        self.clean_prob = clean_prob
        self.shift_prob = shift_prob
        self.erode_prob = erode_prob
        self.dilate_prob = dilate_prob
        self.max_shift = max_shift
        self.morph_kernel = morph_kernel
        self.morph_iters = morph_iters
        self.require_foreground = require_foreground

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        image = image.float()

        clean = (image > 0).float()

        if self.require_foreground and not torch.any(clean > 0):
            return clean

        if torch.rand(1).item() < self.clean_prob:
            return clean

        if torch.rand(1).item() >= self.prob:
            return clean

        x = clean.clone()

        if torch.rand(1).item() < self.shift_prob:
            x = random_shift3d(x, max_shift=self.max_shift)

        if torch.rand(1).item() < self.erode_prob:
            n_iter = np.random.randint(self.morph_iters[0], self.morph_iters[1] + 1)
            x = binary_erode3d(x, kernel_size=self.morph_kernel, iterations=n_iter)

        if torch.rand(1).item() < self.dilate_prob:
            n_iter = np.random.randint(self.morph_iters[0], self.morph_iters[1] + 1)
            x = binary_dilate3d(x, kernel_size=self.morph_kernel, iterations=n_iter)

        return (x > 0.5).float()


def compose_transforms(cf):
    transforms_list = []

    if 'MinMaxNormalize' not in cf['pre_processing']:
        cf['pre_processing']['MinMaxNormalize'] = False

    if cf['pre_processing']['normalize']:
        if cf['pre_processing']['MinMaxNormalize']:
            transforms_list.append(
                MinMaxNormalize(
                    0, 1,
                    clip=cf['pre_processing']['clip'],
                    clip_percent=cf['pre_processing']['clip_percent']
                )
            )
        else:
            transforms_list.append(NaturalNormalize(MAX_PIXEL_VAL=256))

    # Standard augmentation controlled by train_augment / valid_augment
    use_standard_aug = (
        (cf.get('train', False) and cf['pre_processing'].get('train_augment', False)) or
        ((not cf.get('train', False)) and cf['pre_processing'].get('valid_augment', False))
    )

    if use_standard_aug:
        transforms_list.append(
            StandardAugmentation3D(
                prob=cf['pre_processing'].get('augment_prob', 0.5),
                zoom_prob=cf['pre_processing'].get('zoom_prob', 0.3),
                zoom_min=cf['pre_processing'].get('zoom_min', 1.03),
                zoom_max=cf['pre_processing'].get('zoom_max', 1.10),
                rotate_prob=cf['pre_processing'].get('rotate_prob', 0.0),
                rotate_min_deg=cf['pre_processing'].get('rotate_min_deg', -5.0),
                rotate_max_deg=cf['pre_processing'].get('rotate_max_deg', 5.0),
                translate_prob=cf['pre_processing'].get('translate_prob', 0.0),
                translate_max_shift=cf['pre_processing'].get('translate_max_shift', 3),
                require_foreground=cf['pre_processing'].get('require_foreground', True),
            )
        )

    # Post-DAE corruption remains train-only
    use_post_dae = cf['pre_processing'].get('post_dae_train', False) and cf.get('train', False)
    if use_post_dae:
        transforms_list.append(
            PostDAECorruption3D(
                prob=cf['pre_processing'].get('post_dae_prob', 0.5),
                clean_prob=cf['pre_processing'].get('post_dae_clean_prob', 0.5),
                shift_prob=cf['pre_processing'].get('post_dae_shift_prob', 0.3),
                erode_prob=cf['pre_processing'].get('post_dae_erode_prob', 0.2),
                dilate_prob=cf['pre_processing'].get('post_dae_dilate_prob', 0.2),
                max_shift=cf['pre_processing'].get('post_dae_max_shift', 3),
                morph_kernel=cf['pre_processing'].get('post_dae_morph_kernel', 3),
                morph_iters=tuple(cf['pre_processing'].get('post_dae_morph_iters', [1, 1])),
                require_foreground=cf['pre_processing'].get('require_foreground', True),
            )
        )

    if cf['dim'] == 2:
        transforms_list.append(v2.ToImage())
        transforms_list.append(v2.Resize(cf['ori_size']))
        transforms_list.append(v2.ToDtype(torch.float32))
        return v2.Compose(transforms_list)

    elif cf['dim'] == 3:
        transforms_list.append(ToDtype_3d(dt='float32'))
        return v2.Compose(transforms_list)