import argparse
import numpy as np
import os
import probreg
from probreg import cpd

from config import get_sample_info
from ir_processing import visualise_ir_matrix


def register_ir_matrix(ir_matrix, hne_coordinates, spectral_coordinates):
    H, W, C = ir_matrix.shape

    print("Registration process started...")
    tf_param, _, _ = cpd.registration_cpd(spectral_coordinates, hne_coordinates)

    # Get all non-background (0 or NaN) coordinates from the original IR matrix
    #all_spectral_coordinates = np.argwhere(~np.isnan(ir_matrix[:,:,0]))
    all_spectral_coordinates = np.argwhere(np.any(ir_matrix != 0, axis=2)) # used if using 0 instead of NaN
    all_spectral_coordinates = all_spectral_coordinates[:, [1, 0]] # switch [y, x] to [x, y]

    transformed_coordinates = tf_param.transform(all_spectral_coordinates)

    max_x = int(np.ceil(np.max(transformed_coordinates[:, 0])))
    max_y = int(np.ceil(np.max(transformed_coordinates[:, 1])))
    new_W = max(W, max_x + 1)
    new_H = max(H, max_y + 1)

    # Transform IR data matrix
    #transformed_ir_matrix = np.ones((new_H, new_W, C)) * np.nan
    transformed_ir_matrix = np.zeros((new_H, new_W, C))
    for (orig_x, orig_y), (new_x, new_y) in zip(all_spectral_coordinates, transformed_coordinates):
        new_x, new_y = int(round(new_x)), int(round(new_y))
        if 0 <= new_x < new_W and 0 <= new_y < new_H:
            transformed_ir_matrix[new_y, new_x, :] = ir_matrix[orig_y, orig_x, :]
    
    print("Registration process completed.")
    return transformed_ir_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_id", choices=['A', 'B', 'C', 'D'])
    args = parser.parse_args()

    input_dir = f"outputs/sample_{args.sample_id}/step1"
    output_dir = f"outputs/sample_{args.sample_id}/step2"
    os.makedirs(output_dir, exist_ok=True)

    ir_matrix = np.load(os.path.join(input_dir, "ir_matrix.npy"))
    sample_info = get_sample_info(args.sample_id)
    
    registered_ir_matrix = register_ir_matrix(ir_matrix, sample_info["hne_coordinates"], sample_info["spectral_coordinates"])
    np.save(os.path.join(output_dir, "registered_ir_matrix.npy"), registered_ir_matrix)

    visualise_ir_matrix(registered_ir_matrix, f"Sample {args.sample_id} Registered IR Image", output_dir, plot=False)


if __name__ == "__main__":
    main()
