import numpy as np
import os
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class IRDataset(Dataset):
    def __init__(self, base_dir, mode, sample='D'):
        super().__init__()
        self.base_dir = base_dir
        self.mode = mode
        self.sample = sample
        
        print(f"Initialising {self.mode} dataset for sample {self.sample}")
        self.data_pairs = self._load_data_pairs()
        
        #print(f"{self.mode.capitalize()} set: Using {len(self.data_pairs)} patches from sample {self.sample}")
        
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
        
        # Load IR data 20x20x801
        ir_spectra = np.load(ir_path)
        ir_spectra = np.transpose(ir_spectra, (2, 0, 1))  # reshape to 801x20x20
        
        # Sample & location (unique patch id)
        image_name = os.path.basename(hne_path)
        patch_id = image_name[0:-8]
        
        return patch_id, ir_spectra, hne_image

    # Example patch unique id: C_51x91; Example filename: C_51x91_HNE.png
    def _load_data_pairs(self):
        data_pairs = []
        
        sample_dir = os.path.join(self.base_dir, self.sample)
        mode_dir = os.path.join(sample_dir, self.mode)
        
        ir_data_dir = os.path.join(mode_dir, "standardised", "ir_data")        
        hne_patches_dir = os.path.join(mode_dir, "hne_patches")
                
        hne_files = [f for f in os.listdir(hne_patches_dir) if f.endswith("_HNE.png")]
        
        for hne_file in hne_files:
            base_name = hne_file.replace("_HNE.png", '')
            ir_file = f"{base_name}_IR.npy"
            ir_path = os.path.join(ir_data_dir, ir_file)
            
            data_pairs.append((os.path.join(hne_patches_dir, hne_file), ir_path))
        
        return data_pairs

    def _read_image(self, image_path):
        image = Image.open(image_path)
        return np.asarray(image)
