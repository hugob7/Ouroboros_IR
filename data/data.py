import glob
import os
import pandas as pd
import sys
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from scipy.spatial import distance
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import tifffile
import configparser
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from combat.pycombat import pycombat
from utils import DEVICE
import joblib
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# IR Dataset class currently using full IR patch data 20x20x801
# Can be modified to work with 801d vector or 15d vector from PCA too
class IRDataset(Dataset):
    def __init__(self, base_dir, mode):
        super(Dataset, self).__init__()
        self.base_dir = base_dir
        self.mode = mode
        self.samples = ['A', 'B', 'C', 'D']

        if self.mode == 'train':
            self.samples = self.samples[:3]  # Training: A, B, C
        else:
            self.samples = self.samples[3:]  # Testing: D
        
        self.data_pairs = self._load_data_pairs()

    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, index):
        hne_path, ir_path = self.data_pairs[index]
        
        # Load H&E patch image (256x256x3)
        hne_image = self._read_image(hne_path)
        hne_image = hne_image / 255.0
        hne_image = hne_image[:, :, :3]
        transform = T.Compose([T.ToTensor()])
        hne_image = transform(hne_image)
        
        # Load corresponding IR patch data (20x20x801, already min-max normalised)
        ir_spectra = np.load(ir_path)
        ir_spectra = np.transpose(ir_spectra, (2, 0, 1)) # reshape 801x20x20

        #print(ir_spectra.shape)
        
        # Sample & location (unique patch id)
        image_name = os.path.basename(hne_path)
        patch_id = image_name[0:-8]
        
        return patch_id, ir_spectra, hne_image
    
    def _load_data_pairs(self):
        data_pairs = []
        for sample in self.samples:
            hne_dir = os.path.join(self.base_dir, sample, 'hne_patches_new')
            ir_dir = os.path.join(self.base_dir, sample, 'ir_data_full_normalised')
            
            for hne_file in os.listdir(hne_dir):
                if hne_file.endswith('_HNE.png'):
                    base_name = hne_file.replace('_HNE.png', '')
                    ir_file = f'{base_name}_IR_full.npy'
                    if os.path.exists(os.path.join(ir_dir, ir_file)):
                        data_pairs.append((
                            os.path.join(hne_dir, hne_file),
                            os.path.join(ir_dir, ir_file)
                        ))
        return data_pairs

    def _read_image(self,image_path):
        image = Image.open(image_path)
        return np.asarray(image)
