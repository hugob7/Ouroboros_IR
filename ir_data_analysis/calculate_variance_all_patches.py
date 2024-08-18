import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from tqdm import tqdm


# Load 20z20z801 IR patches
def load_ir_patches(base_dir):
    ir_patches = []
    for sample in ['A', 'B', 'C', 'D']:
        sample_dir = os.path.join(base_dir, sample, 'ir_data_full_new')
        for filename in tqdm(os.listdir(sample_dir), desc=f'Loading sample {sample}'):
            if filename.endswith('_IR_full.npy'):
                file_path = os.path.join(sample_dir, filename)
                patch = np.load(file_path)
                ir_patches.append(patch)
    return ir_patches

# Calculate variance for each wavenumber across all 20z20x801 patches
def calculate_ir_variance(ir_patches):
    ir_data = np.stack(ir_patches, axis=0)
    variances = np.var(ir_data, axis=0)
    return variances

def visualise_ir_variance(variances):
    # Mean variance across spatial dimensions
    mean_variance = np.mean(variances, axis=(0, 1))
    plt.figure(figsize=(12, 6))
    plt.plot(range(801), mean_variance)
    plt.title('Mean Variance Across Spatial Dimensions for Each Wavenumber')
    plt.xlabel('Wavenumber Index')
    plt.ylabel('Mean Variance')
    plt.savefig('mean_variance_by_wavenumber.png')
    plt.close()
    
    # Heatmap mean var across Wvn's
    mean_spatial_variance = np.mean(variances, axis=2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(mean_spatial_variance, cmap='viridis')
    plt.title('Mean Variance Across Wavenumbers for Each Spatial Position')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('spatial_variance_heatmap.png')
    plt.close()
    
    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(variances.flatten(), bins=100)
    plt.title('Distribution of Variances')
    plt.xlabel('Variance')
    plt.ylabel('Frequency')
    plt.savefig('variance_distribution.png')
    plt.close()

#def main():
#    base_dir = '/Users/hugoboland/cs310_project2/patches'
#    
#    print('Loading IR patches...')
#    ir_patches = load_ir_patches(base_dir)
#    print(f'Loaded {len(ir_patches)} patches.')
#    
#    variances = calculate_ir_variance(ir_patches)    
#    visualise_ir_variance(variances)
#    
#    print('Stats:')
#    print(f'Mean variance: {np.mean(variances)}')
#    print(f'Median variance: {np.median(variances)}')
#    print(f'Max variance: {np.max(variances)}')
#    print(f'Min variance: {np.min(variances)}')
#    
#    print('Saving overall variance array..')
#    np.save('ir_data_variances.npy', variances)
#
#    print('Complete')

def load_and_process_patches(base_dir):
    all_values = []
    
    for sample in ['A', 'B', 'C', 'D']:
        sample_dir = os.path.join(base_dir, sample, 'ir_data_full_new')
        for filename in tqdm(os.listdir(sample_dir), desc=f'Processing sample {sample}'):
            if filename.endswith('_IR_full.npy'):
                file_path = os.path.join(sample_dir, filename)
                patch = np.load(file_path)
                all_values.extend(patch.flatten())
    
    return np.array(all_values)

def calculate_stats(values):
    min_value = np.min(values)
    max_value = np.max(values)
    mean_value = np.mean(values)
    
    return min_value, max_value, mean_value

def main():
    base_dir = '/Users/hugoboland/cs310_project2/patches'
    
    print('Loading & processing IR patches...')
    all_values = load_and_process_patches(base_dir)
    
    print('Calculating stats...')
    min_val, max_val, mean_val = calculate_stats(all_values)
    
    print(f'Min value across all spectra: {min_val}')
    print(f'Max value across all spectra: {max_val}')
    print(f'Mean value across all spectra: {mean_val}')

if __name__ == '__main__':
    main()
