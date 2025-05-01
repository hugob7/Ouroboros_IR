import numpy as np
import torch
from data.data import IRDataset
from metrics import calculate_fid_score
from torch.utils.data import DataLoader
from utils import DEVICE


def calculate_noise_fid(dataloader):
    real_images = []
    
    with torch.no_grad():
        for _, (_, _, image_gt) in enumerate(dataloader):
            real_images.append(image_gt.cpu().numpy())
        
    noise_images = []
    for batch in real_images:
        noise_batch = np.random.uniform(low=0, high=1, size=batch.shape).astype(np.float32)
        noise_images.append(noise_batch)
    
    fid_score = calculate_fid_score(real_images, noise_images)
    return fid_score


if __name__ == "__main__":
    base_dir = "./data/patches"
    test_dataset = IRDataset(base_dir, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    fid_score = calculate_noise_fid(test_loader)
    print(f"FID score (random noise images): {fid_score:.2f}")
