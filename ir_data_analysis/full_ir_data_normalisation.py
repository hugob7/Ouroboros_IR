import numpy as np
import os
from tqdm import tqdm


def load_and_process_data(base_path):
    all_data = []
    file_paths = []

    for sample in ['A', 'B', 'C', 'D']:
        sample_path = os.path.join(base_path, sample, 'ir_data_full_new')
        for file in os.listdir(sample_path):
            if file.endswith('_IR_full.npy'):
                file_path = os.path.join(sample_path, file)
                data = np.load(file_path)
                all_data.append(data)
                file_paths.append(file_path)
    
    return np.concatenate(all_data), file_paths

#def min_max_normalise(data, global_min, global_max):
#    return (data - global_min) / (global_max - global_min)

# Ignore background 0 values
def min_max_normalise(data, global_min, global_max):
    non_background_mask = data != 0

    # Normalise only non-background points
    normalised_data = np.zeros_like(data)
    normalised_data[non_background_mask] = (data[non_background_mask] - global_min) / (global_max - global_min)

    return normalised_data

def create_normalised_directory(base_path, sample):
    normalised_dir = os.path.join(base_path, sample, 'ir_data_full_normalised')
    os.makedirs(normalised_dir, exist_ok=True)
    return normalised_dir

def main(base_path):
    print('Loading & concatenating all data..')
    all_data, file_paths = load_and_process_data(base_path)

    print(len(file_paths))

    global_min = np.min(all_data)
    global_max = np.max(all_data)

    print(f'Global min: {global_min}, Global max: {global_max}')

    print('Normalising & saving data...')
    for file_path in tqdm(file_paths):
        data = np.load(file_path)
        normalised_data = min_max_normalise(data, global_min, global_max)
        
        _, filename = os.path.split(file_path)
        sample = filename[0]
        normalised_dir = create_normalised_directory(base_path, sample)
        new_file_path = os.path.join(normalised_dir, filename)
        
        np.save(new_file_path, normalised_data)

    print('Normalisation complete.')

if __name__ == '__main__':
    base_path = '/Users/hugoboland/cs310_project2/patches/'
    main(base_path)