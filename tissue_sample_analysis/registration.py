import numpy as np
import probreg
from probreg import cpd


def register_ir_matrix(IR_matrix, hne_coordinates, spectral_coordinates):
    H, W, C = IR_matrix.shape

    print('Registration process started.')
    tf_param, _, _ = cpd.registration_cpd(spectral_coordinates, hne_coordinates)

    # Get all non-NaN coordinates from the original IR matrix
    all_spectral_coordinates = np.argwhere(~np.isnan(IR_matrix[:,:,0]))
    #all_spectral_coordinates = np.argwhere(np.any(IR_matrix != 0, axis=2)) # used if using 0 instead of NaN
    all_spectral_coordinates = all_spectral_coordinates[:, [1, 0]] # switch [y, x] to [x, y]

    transformed_coordinates = tf_param.transform(all_spectral_coordinates)

    max_x = int(np.ceil(np.max(transformed_coordinates[:, 0])))
    max_y = int(np.ceil(np.max(transformed_coordinates[:, 1])))

    new_W = max(W, max_x + 1)
    new_H = max(H, max_y + 1)
    
    print(f'Original dimensions: {W}x{H}')
    print(f'New dimensions: {new_W}x{new_H}')

    # Transform IR matrix
    transformed_IR = np.ones((new_H, new_W, C)) * np.nan
    #transformed_IR = np.zeros((new_H, new_W, C))
    for (orig_x, orig_y), (new_x, new_y) in zip(all_spectral_coordinates, transformed_coordinates):
        new_x, new_y = int(round(new_x)), int(round(new_y))
        if 0 <= new_x < new_W and 0 <= new_y < new_H:
            transformed_IR[new_y, new_x, :] = IR_matrix[orig_y, orig_x, :]
    
    print('Registration process completed.')
    return transformed_IR
