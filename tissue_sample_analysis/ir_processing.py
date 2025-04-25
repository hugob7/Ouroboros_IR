import argparse
import mat73
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
from config import get_sample_info, BASE_DIR
from PIL import Image


# Create an MxNx801 IR data matrix for the given sample
def create_ir_matrix(mat_file_path):
    data_dict = mat73.loadmat(mat_file_path)
    sample_data = list(data_dict.values())[0]
    outliers_indices = sample_data["outliers_indices"][0].astype("int") - 1
    
    data_matrix = np.squeeze(sample_data["Data"])
    Wvn = sample_data["Wvn"]
    xy = sample_data["xy"]
    C, N = data_matrix.shape
    W, H = int(xy[0,0]), int(xy[0,1])
    
    assert C == 801, f"Expected 801 wavenumbers, but got {C}"
    assert N == (H * W - len(outliers_indices)), "Mismatch in data dimensions"

    # Create the MxNx801 matrix - can switch to NaN if wanted
    #ir_matrix = np.ones((H, W, C)) * np.nan
    ir_matrix = np.zeros((H, W, C))
    mask = np.ones(W * H, dtype=bool)
    mask[outliers_indices] = False
    
    for c in range(C):
        #Z = np.ones(W * H) * np.nan
        Z = np.zeros(W * H)
        Z[mask] = data_matrix[c]
        ir_matrix[:, :, c] = np.reshape(Z, (H, W))
    
    # Flip IR matrix to correct orientation
    ir_matrix = np.flipud(ir_matrix)

    #spectral_coordinates = np.argwhere(~np.isnan(ir_matrix[:, :, 0]))
    spectral_coordinates = np.argwhere(~np.all(ir_matrix == 0, axis=2))
    
    return ir_matrix, spectral_coordinates, Wvn


def visualise_ir_matrix(ir_matrix, title, output_dir, plot=False):
    if plot:
        ir_vis = ir_matrix.copy()
        ir_vis[ir_vis == 0] = np.nan
        avg_ir = np.nanmean(ir_vis, axis=2)

        plt.figure(figsize=(10, 8))
        plt.imshow(avg_ir, cmap="gray")
        plt.axis("off")
        plt.colorbar()
        plt.title(title)
        plt.savefig(os.path.join(output_dir, "ir_matrix_image.png"))
        plt.show()
    else:
        bm = np.sum(ir_matrix, axis=2) == 0
        image = np.ones(ir_matrix.shape[:2], dtype=np.uint8) * 255
        image[bm] = 0
        image = Image.fromarray(image, mode='L')
        image.save(os.path.join(output_dir, "ir_image.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_id", choices=['A', 'B', 'C', 'D'])
    args = parser.parse_args()

    print(f"Creating IR data matrix for sample {args.sample_id}...")

    sample_info = get_sample_info(args.sample_id)
    mat_file_path = os.path.join(BASE_DIR, sample_info["file"])
    output_dir = f"outputs/sample_{args.sample_id}/step1"
    os.makedirs(output_dir, exist_ok=True)

    ir_matrix, spectral_coordinates, wvn = create_ir_matrix(mat_file_path)

    np.save(os.path.join(output_dir, "ir_matrix.npy"), ir_matrix)
    np.save(os.path.join(output_dir, "spectral_coordinates.npy"), spectral_coordinates)
    np.save(os.path.join(output_dir, "wvn.npy"), wvn)

    print("Finished creating IR data matrix.")
    visualise_ir_matrix(ir_matrix, f"Sample {args.sample_id} IR Image", output_dir, plot=False)


if __name__ == "__main__":
    main()
