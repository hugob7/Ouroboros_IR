import argparse
import json
import numpy as np
import os
import random
import re
import shutil
from sklearn.preprocessing import StandardScaler


def extract_patch_coordinates(sample, ir_data_dir):
    patch_coords = []
    for filename in os.listdir(ir_data_dir):
        if filename.endswith("_IR.npy"):
            # Extract unique coordinates part of filename (ex: 51x91 from A_51x91_IR.npy)
            match = re.match(f"{sample}_(.+)_IR.npy", filename)
            patch_coords.append(match.group(1))
    return patch_coords


def split_patches(patch_coords, train_pct):
    random.shuffle(patch_coords)
    
    train_count = int(len(patch_coords) * train_pct / 100)
    train_patches = patch_coords[:train_count]
    test_patches = patch_coords[train_count:]
    
    return train_patches, test_patches


# Create train and test subdirectories 
def create_directory_structure(sample_dir, standardise=True):
    train_dir = os.path.join(sample_dir, "train")
    test_dir = os.path.join(sample_dir, "test")
    
    for subdir in ["ir_data", "hne_patches", "ir_patches"]:
        os.makedirs(os.path.join(train_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(test_dir, subdir), exist_ok=True)
        
        if standardise and subdir == "ir_data":
            os.makedirs(os.path.join(train_dir, "standardised", subdir), exist_ok=True)
            os.makedirs(os.path.join(test_dir, "standardised", subdir), exist_ok=True)
    
    return train_dir, test_dir


def copy_files(sample, src_dir, dst_dir, patch_coords, file_ext):
    copied_files = []
    
    for coord in patch_coords:
        src_file = os.path.join(src_dir, f"{sample}_{coord}{file_ext}")
        dst_file = os.path.join(dst_dir, f"{sample}_{coord}{file_ext}")
        shutil.copy2(src_file, dst_file)
        copied_files.append(dst_file)

    return copied_files


def collect_valid_train_spectra(train_files):
    all_train_spectra = []
    
    for ir_file in train_files:
        patch_data = np.load(ir_file)
        M, N, num_wavenumbers = patch_data.shape

        patch_2d = patch_data.reshape(-1, num_wavenumbers)        
        valid_spectra = patch_2d[~np.all(patch_2d == 0, axis=1)]
        
        all_train_spectra.append(valid_spectra)
    
    return np.vstack(all_train_spectra)


def standardise_patch(ir_file, output_file, scaler):
    patch_data = np.load(ir_file)
    M, N, num_wavenumbers = patch_data.shape
    
    std_patch = patch_data.copy()
    std_patch_2d = std_patch.reshape(-1, num_wavenumbers)
    
    non_background_mask = ~np.all(std_patch_2d == 0, axis=1)
    std_patch_2d[non_background_mask] = scaler.transform(std_patch_2d[non_background_mask])    
    std_patch = std_patch_2d.reshape(M, N, num_wavenumbers)
    
    np.save(output_file, std_patch)


# For a sample: split patches, copy files, standardise
def process_sample(sample, base_dir, train_pct, standardise):    
    sample_dir = os.path.join(base_dir, sample)
    ir_data_dir = os.path.join(sample_dir, "ir_data")
    
    patch_coords = extract_patch_coordinates(sample, ir_data_dir)
    
    train_patches, test_patches = split_patches(patch_coords, train_pct)
    
    print(f"Total patches: {len(patch_coords)}")
    print(f"Training patches: {len(train_patches)} ({train_pct}%)")
    print(f"Test patches: {len(test_patches)} ({100 - train_pct}%)")
    
    train_dir, test_dir = create_directory_structure(sample_dir, standardise)

    datasets = {"training": (train_dir, train_patches), "test": (test_dir, test_patches)}
    
    for subdir in ["ir_data", "hne_patches", "ir_patches"]:
        src_dir = os.path.join(sample_dir, subdir)   
        ext = "_IR.npy" if subdir == "ir_data" else "_HNE.png" if subdir == "hne_patches" else "_IR.png" if subdir == "ir_patches" else None

        copy_files(sample, src_dir, os.path.join(train_dir, subdir), train_patches, ext)
        copy_files(sample, src_dir, os.path.join(test_dir, subdir), test_patches, ext)
    
    if standardise:
        print(f"Standardising IR spectra for sample {sample}...")
        
        train_ir_files = [os.path.join(train_dir, "ir_data", f"{sample}_{coord}_IR.npy") for coord in train_patches]
        all_train_spectra = collect_valid_train_spectra(train_ir_files)
        
        # Fit StandardScaler on all valid train spectra (non-background)
        scaler = StandardScaler()
        scaler.fit(all_train_spectra)

        # Apply standardisation to train and test spectra
        for dataset_name, (dataset_dir, coords) in datasets.items():
            print(f"Standardising {dataset_name} set...")
            
            for coord in coords:
                ir_file = os.path.join(dataset_dir, "ir_data", f"{sample}_{coord}_IR.npy")
                std_file = os.path.join(dataset_dir, "standardised", "ir_data", f"{sample}_{coord}_IR.npy")
                standardise_patch(ir_file, std_file, scaler)
        
        print(f"Finished standardising spectra for sample {sample}")


# Split IR spectra patches into train & independent test set, and standardise spectra after
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./data/patches")
    parser.add_argument("--train_pct", type=int, default=30)
    #parser.add_argument("--standardise", action='store_true')
    args = parser.parse_args()

    for sample in ['A', 'B', 'C', 'D']:
        print(f"\nProcessing Sample {sample}...")
        process_sample(sample, args.base_dir, args.train_pct, standardise=True)


if __name__ == "__main__":
    main()
