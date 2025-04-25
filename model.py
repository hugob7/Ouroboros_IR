import torch.nn as nn
from generators import tissue_image_generator
from utils import DEVICE

class GenerativeModel(nn.Module):
    def __init__(self, mode='train',
                 normalization='instance', activation='leakyrelu-0.2', generator_name='dcgan',
                 **kwargs):
        super(GenerativeModel, self).__init__()
        self.mode = mode
        input_channels = 801 # 801 wavenumbers for spectra
        self.image_generator = tissue_image_generator(input_dim=input_channels,
                                                      output_nc=3,
                                                      generator_name=generator_name,
                                                      n_blocks_global=2,
                                                      n_downsample_global=3,
                                                      ngf=64,
                                                      norm='instance')
        self.image_generator.to(DEVICE)

    def forward(self, ir_features):
        generated_image = self.image_generator(ir_features)
        return generated_image
