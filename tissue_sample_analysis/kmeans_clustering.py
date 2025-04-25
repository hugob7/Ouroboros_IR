import argparse
import numpy as np
import os
from PIL import Image
from sklearn.cluster import KMeans


def create_kmeans_clustering(IR_matrix, n_clusters=4):
    H, W, C = IR_matrix.shape
    
    # Prepare data for k-means clustering
    IR_2d = IR_matrix.reshape(-1, C).astype(np.float32)
    valid_indices = ~np.isnan(IR_2d).any(axis=1)
    IR_2d_valid = IR_2d[valid_indices]
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, verbose=1)
    kmeans.fit(IR_2d_valid)
    labels = kmeans.labels_
    
    label_matrix = np.ones(H*W, dtype=int) * -1
    label_matrix[valid_indices] = labels
    label_matrix = label_matrix.reshape(H, W)

    onehot_matrix = np.zeros((H, W, n_clusters))
    for i in range(n_clusters):
        onehot_matrix[:, :, i] = (label_matrix == i).astype(float)
    
    cluster_colours = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
    colour_map = np.array(cluster_colours, dtype=np.uint8)
    rgb_image = colour_map[label_matrix + 1]
    
    return Image.fromarray(rgb_image, "RGB"), onehot_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_id", choices=['A', 'B', 'C', 'D'])
    args = parser.parse_args()

    print(f"Performing K-means clustering for sample {args.sample_id}...")

    input_dir = f"outputs/sample_{args.sample_id}/step2"
    output_dir = f"outputs/sample_{args.sample_id}/step3"
    os.makedirs(output_dir, exist_ok=True)

    registered_ir_matrix = np.load(os.path.join(input_dir, "registered_ir_matrix.npy"))

    clustered_ir_image, clustered_ir_matrix = create_kmeans_clustering(registered_ir_matrix)

    clustered_ir_image.save(os.path.join(output_dir, "clustered_ir_image.png"))
    np.save(os.path.join(output_dir, "clustered_ir_matrix.npy"), clustered_ir_matrix)

    print("Finished K-means clustering.")


if __name__ == "__main__":
    main()
