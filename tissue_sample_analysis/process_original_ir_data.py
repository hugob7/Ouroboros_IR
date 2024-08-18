import mat73
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io


# Create an MxNx801 IR matrix for a given core
def create_ir_matrix(mat_file_path):
    if os.path.basename(mat_file_path) == 'n935_19b1_EMSC.mat': # Core B
        data_dict = scipy.io.loadmat(mat_file_path, simplify_cells=True)
        core_data = data_dict['n935_19b1_EMSC']
        outliers_indices = core_data['outliers_indices'].astype('int') - 1
    else:
        data_dict = mat73.loadmat(mat_file_path)
        core_data = list(data_dict.values())[0]
        outliers_indices = core_data['outliers_indices'][0].astype('int') - 1
    
    # Extract core-specific data (assuming there's only one key in the dict)
    data_matrix = core_data['Data']
    Wvn = core_data['Wvn']
    xy = core_data['xy']
    C, N = data_matrix.shape
    W, H = int(xy[0,0]), int(xy[0,1])
    
    assert C == 801, f'Expected 801 wavenumbers, but got {C}'
    assert N == (H * W - len(outliers_indices)), 'Mismatch in data dimensions'

    # Create the MxNx801 matrix
    IR_matrix = np.ones((H, W, C)) * np.nan
    mask = np.ones(W * H, dtype=bool)
    mask[outliers_indices] = False
    
    for c in range(C):
        Z = np.ones(W * H) * np.nan
        Z[mask] = data_matrix[c]
        IR_matrix[:, :, c] = np.reshape(Z, (H, W))
    
    # Flip IR matrix to correct orientation
    IR_matrix = np.flipud(IR_matrix)
    
    # Get spectral coordinates
    spectral_coordinates = np.argwhere(~np.isnan(IR_matrix[:, :, 0]))
    
    return IR_matrix, spectral_coordinates, Wvn


# Visualise average of IR matrix
def plot_ir_matrix(IR_matrix, title):
    avg_IR = np.nanmean(IR_matrix, axis=2)
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_IR, cmap='gray')
    plt.colorbar()
    plt.title(title)
    plt.show()
    