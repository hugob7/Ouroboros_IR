import numpy as np
from sklearn.cluster import KMeans
from PIL import Image


def create_kmeans_ir_image(IR_matrix, n_clusters=4):
    H, W, C = IR_matrix.shape
    
    IR_2d = IR_matrix.reshape(-1, C).astype(np.float32)
    
    valid_indices = ~np.isnan(IR_2d).any(axis=1)
    IR_2d_valid = IR_2d[valid_indices]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(IR_2d_valid)
    labels = kmeans.labels_
    
    clustered_image = np.ones(H*W, dtype=int) * -1
    clustered_image[valid_indices] = labels
    clustered_image = clustered_image.reshape(H, W)
    
    cluster_colours = [
        (0, 0, 0),
        (255, 255, 255),
        (192, 192, 192),
        (128, 128, 128),
        (224, 224, 224)
    ]
    colour_map = np.array(cluster_colours, dtype=np.uint8)
    
    rgb_image = colour_map[clustered_image + 1]
    
    return Image.fromarray(rgb_image, 'RGB'), clustered_image
