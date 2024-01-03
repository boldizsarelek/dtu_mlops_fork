import torch
import os
import fnmatch
from torch.utils.data import Dataset

def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find speech files recursively.
        Args:
            root_dir (str): Root root_dir to find.
            query (str): Query to find.
            include_root_dir (bool): If False, root_dir name is not included.
        Returns:
            list: List of found filenames.
    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files

def mnist():
    """Return train and test dataloaders for MNIST."""
    project_dir = "/Users/boldizsarelek/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/MLOps/dtu_mlops"
    data_dir = os.path.join(project_dir, "data", "corruptmnist")
    
    train_img_files = find_files(data_dir, query="train_images*.pt")
    train_target_files = find_files(data_dir, query="train_target*.pt")
    test_img_files = find_files(data_dir, query="test_images*.pt")
    test_target_files = find_files(data_dir, query="test_target*.pt")

    train = CustomImageDataset(train_img_files, train_target_files)
    test = CustomImageDataset(test_img_files, test_target_files)

    return train, test

class CustomImageDataset(Dataset):
    def __init__(self, image_files, target_files):
        self.image_files = image_files
        self.target_files = target_files
        self.images=self.load_tensors(image_files)
        self.targets=self.load_tensors(target_files)

        #print(f"shape of images: {self.images.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]
    
    def load_tensors(self, files):
        return torch.cat([torch.load(f) for f in files])
