import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def load_ir_data(base_dir):
    ir_data = []
    file_names = []
    for sample in ['A', 'B', 'C', 'D']:
        sample_dir = os.path.join(base_dir, sample, 'ir_data')
        for filename in tqdm(os.listdir(sample_dir), desc=f'Loading sample {sample}'):
            if filename.endswith('_IR_mean.npy'):
                file_path = os.path.join(sample_dir, filename)
                ir_vector = np.load(file_path)
                ir_data.append(ir_vector)
                file_names.append(os.path.join(sample, filename))
    return np.array(ir_data), file_names

def apply_pca(ir_data, variance_threshold=0.95):
    scaler = StandardScaler()
    ir_data_scaled = scaler.fit_transform(ir_data)
    
    pca = PCA()
    pca.fit(ir_data_scaled)
    
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
    
    return pca, scaler, cumulative_variance_ratio, n_components

def save_pca_data(ir_data_pca, file_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, file_name in enumerate(file_names):
        output_path = os.path.join(output_dir, file_name.replace('_IR_mean.npy', '_IR_PCA.npy'))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, ir_data_pca[i])

def plot_sample_spectra(ir_data, n_samples=10):
    plt.figure(figsize=(12, 6))
    for i in np.random.choice(ir_data.shape[0], n_samples, replace=False):
        plt.plot(ir_data[i])
    plt.title('Sample IR Spectra')
    plt.xlabel('Wavenumber Index')
    plt.ylabel('Intensity')
    plt.savefig('sample_spectra.png')
    plt.close()

def plot_data_distribution(ir_data):
    plt.figure(figsize=(10, 6))
    plt.hist(ir_data.flatten(), bins=100)
    plt.title('Distribution of IR Data Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig('data_distribution.png')
    plt.close()

def plot_explained_variance(pca):
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title('Cumulative Explained Variance Ratio')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.savefig('explained_variance.png')
    plt.close()

def main():
    base_dir = '/Users/hugoboland/cs310_project2/ouroboros/Ouroboros/data/patches'
    output_dir = '/Users/hugoboland/cs310_project2/ouroboros/Ouroboros/data/patches/ir_data_pca'
    variance_threshold = 0.95
    
    print('Loading IR data...')
    ir_data, file_names = load_ir_data(base_dir)

    plot_sample_spectra(ir_data)
    plot_data_distribution(ir_data)
    
    print('Applying PCA..')
    pca, scaler, cumulative_variance_ratio, n_components = apply_pca(ir_data, variance_threshold)

    plot_explained_variance(pca)
    
    #print('Saving PCA-reduced data...')
    #save_pca_data(ir_data_pca, file_names, output_dir)
    
    print(f'Number of components selected: {n_components}')
    print(f'Cumulative explained variance ratio: {cumulative_variance_ratio[-1]:.4f}')
    print(f'PCA-reduced data saved in {output_dir}')
    
if __name__ == '__main__':
    main()