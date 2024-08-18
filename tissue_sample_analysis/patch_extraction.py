import numpy as np
import os
import pandas as pd
import gc
from PIL import Image
from skimage.io import imread, imsave
from sklearn.preprocessing import StandardScaler


def extract_patches(core, registered_IR, protein_expressions_df, core_info, extracted_patches_dir):
    core_dir = os.path.join(extracted_patches_dir, core)
    ir_patch_data_full_dir = os.path.join(core_dir, 'ir_data_full_new')
    ir_patch_data_mean_dir = os.path.join(core_dir, 'ir_data_mean_new')
    ir_patch_img_dir = os.path.join(core_dir, 'ir_patches_new')
    hne_patch_dir = os.path.join(core_dir, 'hne_patches_new')

    for directory in [core_dir, ir_patch_data_full_dir, ir_patch_data_mean_dir, ir_patch_img_dir, hne_patch_dir]:
        os.makedirs(directory, exist_ok=True)

    hne_patch_count = ir_patch_count = 0
    
    image_filename = os.path.join(core_info['images_dir'], f'{core}.png')
    print(f'Processing image file {image_filename} (Extracting patches)')

    I_hne = imread(image_filename)

    protein_expressions_df['image_core'] = protein_expressions_df['VisSpot'].apply(lambda x: x.split('-')[-1])
    protein_expressions_df_current = protein_expressions_df[protein_expressions_df['image_core'] == f'{core}1']

    scaler = StandardScaler()

    for idx, row in protein_expressions_df_current.iterrows():
        y_hne, x_hne = int(row['hne_row_loc']), int(row['hne_col_loc'])
        
        if core in ['B', 'D']: # Cores B & D registered to E-image
            y_e, x_e = int(row['e_row_loc']), int(row['e_col_loc'])
            y_ir, x_ir = int(y_e / core_info['scale_factor_y']), int(x_e / core_info['scale_factor_x'])
        else: # Cores A & C registered to H&E image
            y_ir, x_ir = int(y_hne / core_info['scale_factor_y']), int(x_hne / core_info['scale_factor_x'])
        
        if y_hne == 0 and x_hne == 0:
            continue

        # Extract H&E patch
        if (y_hne - core_info['hw_hne'] >= 0 and y_hne + core_info['hw_hne'] < I_hne.shape[0] and
            x_hne - core_info['hw_hne'] >= 0 and x_hne + core_info['hw_hne'] < I_hne.shape[1]):
            patch_hne = I_hne[y_hne - core_info['hw_hne'] : y_hne + core_info['hw_hne'], 
                             x_hne - core_info['hw_hne'] : x_hne + core_info['hw_hne']]
            patch_name_hne = f"{core}_{row['array_row']}x{row['array_col']}_HNE.png"
            imsave(os.path.join(hne_patch_dir, patch_name_hne), patch_hne)
            hne_patch_count += 1

        # Extract IR patch (all 801 wavenumbers)
        if (y_ir - core_info['hw_ir'] >= 0 and y_ir + core_info['hw_ir'] < registered_IR.shape[0] and 
            x_ir - core_info['hw_ir'] >= 0 and x_ir + core_info['hw_ir'] < registered_IR.shape[1]):
            ir_patch = registered_IR[y_ir - core_info['hw_ir'] : y_ir + core_info['hw_ir'], 
                                     x_ir - core_info['hw_ir'] : x_ir + core_info['hw_ir'], :]
            
            if np.isnan(ir_patch).all():
                print(f"Skipping entirely NaN patch at {core}_{row['array_row']}x{row['array_col']}")
                continue

            if np.isnan(ir_patch).any():
                print(f"NaN values detected in patch at {core}_{row['array_row']}x{row['array_col']}")
            
            # Normalise IR patch
            ir_patch_flat = ir_patch.reshape(-1, ir_patch.shape[-1])
            ir_patch_norm = scaler.fit_transform(ir_patch_flat)
            ir_patch_norm = ir_patch_norm.reshape(ir_patch.shape)

            patch_name_ir_full = f"{core}_{row['array_row']}x{row['array_col']}_IR_full.npy"
            np.save(os.path.join(ir_patch_data_full_dir, patch_name_ir_full), ir_patch_norm)
            
            # Compute and save the mean spectrum (801-vector)
            mean_spectrum = np.nanmean(ir_patch, axis=(0, 1))
            patch_name_ir_mean = f"{core}_{row['array_row']}x{row['array_col']}_IR_mean.npy"
            np.save(os.path.join(ir_patch_data_mean_dir, patch_name_ir_mean), mean_spectrum)
            
            # Also save an average intensity image for visualization
            avg_ir_patch = np.nanmean(ir_patch, axis=2)
            binary_mask = np.where(np.isnan(avg_ir_patch), 0, 255)
            img_ready_ir_patch = binary_mask.astype(np.uint8)
            img_ir_patch = Image.fromarray(img_ready_ir_patch, 'L')
            
            patch_name_ir_avg = f"{core}_{row['array_row']}x{row['array_col']}_IR_avg.png"
            img_ir_patch.save(os.path.join(ir_patch_img_dir, patch_name_ir_avg))
            
            ir_patch_count += 1
    
        # Clear some memory
        del ir_patch, ir_patch_flat, ir_patch_norm, mean_spectrum, avg_ir_patch, binary_mask, img_ready_ir_patch, img_ir_patch
        gc.collect()

        # Optional: Print memory usage every 100 patches
        if (hne_patch_count + ir_patch_count) % 100 == 0:
            print(f"Processed {hne_patch_count + ir_patch_count} patches")

    print(f'Finished extracting patches for core {core}')
    print(f'H&E patch count: {hne_patch_count}')
    print(f'IR patch count: {ir_patch_count}')
