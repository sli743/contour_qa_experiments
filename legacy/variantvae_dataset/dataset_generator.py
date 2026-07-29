import os
import torch
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk
from random import sample


class ProjectionDataset(Dataset):
    def __init__(self, data_path, cf=None, transform=None, sample_ratio=1.0):
        self.proj_dataset_names = []
        self.transform = transform
        self.label_mask = None

        if cf is not None and 'label_mask' in cf and cf['label_mask'] is not None:
            self.label_mask = np.array(cf['label_mask']).astype(int)

        # Collect all nii.gz files
        for root, dirs, files in os.walk(data_path):
            files.sort(key=lambda x: int(x.split('.')[0].split('_')[2]))
            for file in files:
                if file.endswith('.nii.gz'):
                    proj_path = os.path.join(root, file)
                    self.proj_dataset_names.append(proj_path)

        # Filter based on label_mask
        if self.label_mask is not None:
            print(f"\n > Filtering dataset with label_mask = {self.label_mask.tolist()}")
            filtered_names = []

            for path in self.proj_dataset_names:
                img = sitk.GetArrayFromImage(sitk.ReadImage(path))
                unique_labels = np.unique(img)

                # Check if ALL labels in label_mask exist
                if np.all(np.isin(self.label_mask, unique_labels)):
                    filtered_names.append(path)

            self.proj_dataset_names = filtered_names

        # Optional sampling
        if sample_ratio < 1:
            self.proj_dataset_names = sample(
                self.proj_dataset_names,
                int(sample_ratio * len(self.proj_dataset_names))
            )

        self.ids = len(self.proj_dataset_names)
        print(f'\n > Creating dataset with {self.ids} projection images.')

    def __len__(self):
        return self.ids

    def __getitem__(self, idx):
        proj_name = self.proj_dataset_names[idx]
        proj_npy = sitk.GetArrayFromImage(sitk.ReadImage(proj_name))

        # Shape: [1, D, H, W]
        proj_npy = proj_npy.reshape((1, *proj_npy.shape))

        clean_image = torch.from_numpy(proj_npy).float()

        if self.transform:
            proj_image = self.transform(proj_npy)
            if isinstance(proj_image, np.ndarray):
                proj_image = torch.from_numpy(proj_image)
        else:
            proj_image = clean_image.clone()

        return {
            'proj_image': proj_image,      # model input
            'target_image': clean_image,   # clean reconstruction target
            'proj_name': proj_name
        }