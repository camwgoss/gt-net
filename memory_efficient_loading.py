import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gc

class ChunkedLiverTumorDataset(Dataset):
    """
    Memory-efficient dataset that processes data in chunks
    """
    def __init__(self, npz_path, subset='train', downsample_factor=2):
        """
        Args:
            npz_path: Path to NPZ file
            subset: 'train', 'val', or 'test'
            downsample_factor: Factor by which to downsample images (1 = no downsampling)
        """
        self.npz_path = npz_path
        self.subset = subset
        self.downsample_factor = downsample_factor

        # Keys for different subsets
        self.image_key = f'images_{subset}'
        self.mask_key = f'labels_{subset}'

        # Load file info without loading data
        with np.load(npz_path, mmap_mode='r') as data:
            self.length = data[self.image_key].shape[0]
            self.image_shape = data[self.image_key].shape[1:]

            # Calculate downsampled shape
            if downsample_factor > 1:
                self.target_shape = (
                    self.image_shape[0] // downsample_factor,
                    self.image_shape[1] // downsample_factor
                )
            else:
                self.target_shape = self.image_shape

            print(f"Dataset {subset} has {self.length} samples")
            print(f"Original shape: {self.image_shape}, Target shape: {self.target_shape}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Memory-mapped loading - only loads the specific index
        with np.load(self.npz_path, mmap_mode='r') as data:
            # Load just this specific image and mask
            image = data[self.image_key][idx].copy()
            mask = data[self.mask_key][idx].copy()

        # Downsample if needed
        if self.downsample_factor > 1:
            # Simple downsampling by taking every nth pixel
            image = image[::self.downsample_factor, ::self.downsample_factor]
            mask = mask[::self.downsample_factor, ::self.downsample_factor]

        # Add channel dimension if needed
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)

        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        # Normalize image values to [0, 1] if not already
        if image.max() > 1.0:
            image = image / 255.0

        # Ensure mask is binary (0 or 1)
        mask = (mask > 0.5).float()

        return image, mask

def create_dataloaders(data_path, batch_size=8, num_workers=1,
                       downsample_factor=2):
    """
    Create dataloaders with memory efficiency in mind
    """
    # Use memory-efficient dataset
    train_dataset = ChunkedLiverTumorDataset(
        data_path, subset='train', downsample_factor=downsample_factor)
    val_dataset = ChunkedLiverTumorDataset(
            data_path, subset='val', downsample_factor=downsample_factor)

    # When using ChunkedDataset, we can use more workers since we're not loading all at once
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=False)

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=False)

    return train_loader, val_loader

# Example function to examine NPZ file without loading all data
def examine_npz(npz_path):
    """Print information about the NPZ file"""
    loader = ChunkedNpzLoader(npz_path)
    shapes, dtypes = loader.get_info()

    # Try loading a small sample
    print("\nLoading a small sample of the training data:")
    sample_images = loader.load_data_chunk('images_train', 0, 2)
    sample_labels = loader.load_data_chunk('labels_train', 0, 2)

    print(f"Sample images shape: {sample_images.shape}")
    print(f"Sample labels shape: {sample_labels.shape}")
    print(f"Sample images min/max: {sample_images.min()}/{sample_images.max()}")
    print(f"Sample labels min/max: {sample_labels.min()}/{sample_labels.max()}")

    return shapes, dtypes


class ChunkedNpzLoader:
    """
    Memory-efficient loader for large NPZ files.
    Loads data in chunks instead of all at once.
    """

    def __init__(self, npz_path):
        self.npz_path = npz_path
        # Only load the metadata, not the actual arrays
        with np.load(npz_path, mmap_mode='r') as data:
            self.keys = list(data.keys())
            self.shapes = {k: data[k].shape for k in self.keys}
            self.dtypes = {k: data[k].dtype for k in self.keys}

    def get_info(self):
        """Return information about the NPZ file without loading data"""
        print(f"NPZ file: {self.npz_path}")
        print(f"Keys: {self.keys}")
        total_size_gb = 0
        for k in self.keys:
            size_bytes = np.prod(self.shapes[k]) * self.dtypes[k].itemsize
            size_gb = size_bytes / (1024 ** 3)
            total_size_gb += size_gb
            print(f"  {k}: shape {self.shapes[k]}, dtype {self.dtypes[k]}, size {size_gb:.2f} GB")
        print(f"Total size: {total_size_gb:.2f} GB")
        return self.shapes, self.dtypes

    def load_data_chunk(self, key, start_idx=0, end_idx=None, downsample=False):
        """
        Load a specific slice of an array from the NPZ file

        Args:
            key: The key of the array to load
            start_idx: Starting index (along first dimension)
            end_idx: Ending index (along first dimension)
            downsample: If True, downsample images to reduce memory usage

        Returns:
            Numpy array containing the requested slice
        """
        with np.load(self.npz_path, mmap_mode='r') as data:
            if end_idx is None:
                end_idx = self.shapes[key][0]

            # Create indexing slices
            slices = [slice(start_idx, end_idx)]
            for _ in range(1, len(self.shapes[key])):
                slices.append(slice(None))

            # Load the data chunk
            data_chunk = data[key][tuple(slices)].copy()

            # Optionally downsample images to save memory
            if downsample and len(self.shapes[key]) > 2:
                # Assuming data shape is [samples, height, width]
                downsampled = []
                for i in range(data_chunk.shape[0]):
                    # Downsample by taking every other pixel
                    downsampled.append(data_chunk[i, ::2, ::2])
                data_chunk = np.stack(downsampled)

            return data_chunk