import argparse
import os
import pandas as pd

from config import get_sample_info, BASE_DIR, IMAGES_DIR, EXTRACTED_PATCHES_DIR, PROTEIN_EXPRESSIONS_CSV
from ir_processing import create_ir_matrix, visualise_ir_matrix
from registration import register_ir_matrix
from kmeans_clustering import create_kmeans_clustering
from patch_extraction import extract_patches


def process_sample(sample_id, sample_info, protein_expressions_df):
    print(f"Processing Sample {sample_id}...")

    mat_file_path = os.path.join(BASE_DIR, sample_info["file"])
    ir_matrix, spectral_coordinates, Wvn = create_ir_matrix(mat_file_path)

    print(f"IR matrix shape: {ir_matrix.shape}")
    print(f"Number of spectra: {len(spectral_coordinates)}")
    print(f"Wavenumber range: {Wvn[0]} - {Wvn[-1]}")
    
    visualise_ir_matrix(ir_matrix, f"Sample {sample_id} IR Image")

    #clustered_image, clustered_matrix = create_kmeans_clustering(ir_matrix)
    #clustered_image.save(f"kmeans_clustered_sample_{sample_id}.png")

    registered_ir_matrix = register_ir_matrix(ir_matrix, sample_info["hne_coordinates"], spectral_coordinates)
    visualise_ir_matrix(registered_ir_matrix, f"Sample {sample_id} Registered IR Image")

    extract_patches(sample_id, registered_ir_matrix, protein_expressions_df, sample_info, EXTRACTED_PATCHES_DIR)
    
    print(f"Finished processing Sample {sample_id}.")
    print('-' * 30)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_id", choices=['A', 'B', 'C', 'D'])
    args = parser.parse_args()

    protein_expressions_df = pd.read_csv(PROTEIN_EXPRESSIONS_CSV)

    sample_info = get_sample_info(args.sample_id)
    process_sample(args.sample_id, sample_info, protein_expressions_df)


if __name__ == "__main__":
    main()
