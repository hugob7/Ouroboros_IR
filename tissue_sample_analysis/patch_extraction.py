import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imread, imsave

from config import get_sample_info, EXTRACTED_PATCHES_DIR, PROTEIN_EXPRESSIONS_CSV

Image.MAX_IMAGE_PIXELS = 933120000


class PatchExtractor:
    def __init__(self, sample, extracted_patches_dir, sample_info):
        self.sample = sample
        self.base_dir = extracted_patches_dir
        self.sample_info = sample_info
        self.setup_directories()

    def setup_directories(self):
        self.sample_dir = os.path.join(self.base_dir, self.sample)
        self.ir_data_dir = os.path.join(self.sample_dir, "ir_data")
        self.ir_images_dir = os.path.join(self.sample_dir, "ir_patches")
        self.hne_dir = os.path.join(self.sample_dir, "hne_patches")
        
        for directory in [self.sample_dir, self.ir_data_dir, self.ir_images_dir, self.hne_dir]:
            os.makedirs(directory, exist_ok=True)
        
    def get_coordinates(self, row):
        y_hne, x_hne = int(row["hne_row_loc"]), int(row["hne_col_loc"])
        
        if self.sample in ['B', 'D']:
            y_e, x_e = int(row["e_row_loc"]), int(row["e_col_loc"])
            y_ir = int(y_e / self.sample_info["scale_factor_y"])
            x_ir = int(x_e / self.sample_info["scale_factor_x"])
        else:
            y_ir = int(y_hne / self.sample_info["scale_factor_y"])
            x_ir = int(x_hne / self.sample_info["scale_factor_x"])
            
        return (y_ir, x_ir), (y_hne, x_hne)
    
    def extract_ir_patch(self, ir_data, coords, row):
        y_ir, x_ir = coords
        hw = self.sample_info["hw_ir"]
        
        if not self._check_ir_bounds(ir_data, y_ir, x_ir, hw):
            return False
            
        patch = ir_data[y_ir-hw:y_ir+hw, x_ir-hw:x_ir+hw, :]
        
        if self._is_invalid_patch(patch):
            #print(f"Skipping invalid patch at {self.sample}_{row['array_row']}x{row['array_col']}")
            return False
            
        self._save_ir_patch(patch, row)
        return True
    
    def extract_hne_patch(self, hne_image, coords, row):
        y_hne, x_hne = coords
        hw = self.sample_info["hw_hne"]
        
        if not self._check_hne_bounds(hne_image, y_hne, x_hne, hw):
            return False
            
        patch = hne_image[y_hne-hw:y_hne+hw, x_hne-hw:x_hne+hw]
        self._save_hne_patch(patch, row)
        return True
    
    def _check_ir_bounds(self, data, y, x, hw):
        return (y - hw >= 0 and y + hw < data.shape[0] and x - hw >= 0 and x + hw < data.shape[1])
    
    def _check_hne_bounds(self, image, y, x, hw):
        return (y - hw >= 0 and y + hw < image.shape[0] and x - hw >= 0 and x + hw < image.shape[1])
    
    def _is_invalid_patch(self, patch):
        return np.all(patch.sum(axis=2) == 0)

    def _save_ir_patch(self, patch, row):
        patch_name = f"{self.sample}_{row['array_row']}x{row['array_col']}_IR.npy"
        np.save(os.path.join(self.ir_data_dir, patch_name), patch)
        
        image = self._create_patch_image2(patch)
        image_name = f"{self.sample}_{row['array_row']}x{row['array_col']}_IR.png"
        image.save(os.path.join(self.ir_images_dir, image_name))

    def _save_hne_patch(self, patch, row):
        patch_name = f"{self.sample}_{row['array_row']}x{row['array_col']}_HNE.png"
        imsave(os.path.join(self.hne_dir, patch_name), patch)

    def _create_patch_image(self, patch):
        cluster_indices = np.argmax(patch, axis=2)
        has_data = np.any(patch, axis=2)
        cluster_indices[~has_data] = -1
        
        cluster_colours = np.array([
            [0, 0, 0],       # Background
            [255, 0, 0],     # Cluster 0
            [0, 255, 0],     # Cluster 1
            [0, 0, 255],     # Cluster 2
            [255, 0, 255]    # Cluster 3
        ], dtype=np.uint8)
        
        coloured_patch = cluster_colours[cluster_indices + 1]
        return Image.fromarray(coloured_patch)
    
    # def _create_patch_image2(self, patch):
    #     background_mask = np.sum(patch, axis=2) == 0
    #     avg_spectrum = np.mean(patch, axis=2)
    #     scaled = (avg_spectrum * 255).clip(0, 255).astype(np.uint8)
    #     scaled[background_mask] = 0
    #     return Image.fromarray(scaled, mode='L')
    
    def _create_patch_image2(self, patch):
        background_mask = np.sum(patch, axis=2) == 0
        binary_image = np.ones(patch.shape[:2], dtype=np.uint8) * 255
        binary_image[background_mask] = 0
        return Image.fromarray(binary_image, mode='L')
    

def extract_patches(sample_id, ir_data, protein_expressions_df, sample_info, extracted_patches_dir):
    print(f"Extracting patches for Sample {sample_id}...")

    patch_extractor = PatchExtractor(sample_id, extracted_patches_dir, sample_info)

    hne_image = imread(os.path.join(sample_info["images_dir"], f"{sample_id}.png"))
    df_current = protein_expressions_df[protein_expressions_df["VisSpot"].apply(lambda x: x.split('-')[-1]) == f"{sample_id}1"]
    
    hne_count = ir_count = 0
    
    for _, row in df_current.iterrows():
        if row["hne_row_loc"] == 0 and row["hne_col_loc"] == 0:
            continue
            
        ir_coords, hne_coords = patch_extractor.get_coordinates(row)
        
        if patch_extractor.extract_ir_patch(ir_data, ir_coords, row):
            ir_count += 1
            
            if patch_extractor.extract_hne_patch(hne_image, hne_coords, row):
                hne_count += 1
        
        #if (hne_count) % 1000 == 0:
        #    print(f"Processed {hne_count} patches")
    
    print(f"Finished extracting patches for Sample {sample_id}.")
    print(f"H&E patch count: {hne_count}")
    print(f"IR patch count: {ir_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_id", choices=['A', 'B', 'C', 'D'])
    args = parser.parse_args()

    # Change to step3 if performed k-means clustering before
    input_dir = f"outputs/sample_{args.sample_id}/step2"
    registered_ir_matrix = np.load(os.path.join(input_dir, "registered_ir_matrix.npy"))

    protein_expressions_df = pd.read_csv(PROTEIN_EXPRESSIONS_CSV)
    sample_info = get_sample_info(args.sample_id)
    extract_patches(args.sample_id, registered_ir_matrix, protein_expressions_df, sample_info, EXTRACTED_PATCHES_DIR)


if __name__ == "__main__":
    main()
