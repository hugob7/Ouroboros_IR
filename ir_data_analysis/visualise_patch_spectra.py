import numpy as np
import os
import random
from pathlib import Path


base_dir = Path('/Users/hugoboland/cs310_project2/patches')
samples = ['A', 'B', 'C', 'D']

num_patches_per_sample = 2
num_spectra_per_patch = 3
num_wavenumbers_to_display = 10 

def load_random_patches(sample, num_patches):
    sample_dir = base_dir / sample / 'ir_data_full_normalised'
    all_patches = list(sample_dir.glob(f'{sample}_*_IR_full.npy'))
    return random.sample(all_patches, min(num_patches, len(all_patches)))

def display_random_spectra(patch_data, num_spectra, num_wavenumbers):
    for i in range(num_spectra):
        x = random.randint(0, patch_data.shape[0] - 1)
        y = random.randint(0, patch_data.shape[1] - 1)
        spectrum = patch_data[x, y, :]
        print(f'  Random position ({x}, {y}):')
        print(f'    First {num_wavenumbers} wavenumber values: {spectrum[:num_wavenumbers]}')
        print(f'    Min: {spectrum.min():.6f}, Max: {spectrum.max():.6f}, Mean: {spectrum.mean():.6f}')
        print()

for sample in samples:
    print(f'Sample {sample}:')
    random_patches = load_random_patches(sample, num_patches_per_sample)
    
    for patch_file in random_patches:
        print(f'Patch: {patch_file.name}')
        patch_data = np.load(patch_file)
        print(f'  Shape: {patch_data.shape}')
        display_random_spectra(patch_data, num_spectra_per_patch, num_wavenumbers_to_display)
    print()