import functools
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
#from torchsummary import summary
from torchinfo import summary
import torchvision
from utils import DEVICE

import biggan_layers

from layers import GlobalAvgPool, build_cnn, ResnetBlock, get_norm_layer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def tissue_image_generator(input_dim, output_nc, generator_name, ngf, n_downsample_global=3, n_blocks_global=9, norm='instance'):
    norm_layer = get_norm_layer(norm_type=norm)
    if(generator_name == 'residual'):
        netG = ResidualGenerator(input_dim, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif(generator_name == 'dcgan'):
        netG = DCGan(input_dim=input_dim)
    elif(generator_name == 'biggan'):
        netG = BigGanGenerator(dim_z=input_dim)
    elif generator_name == 'pix2pix':
        netG = Pix2PixGenerator(in_channels=input_dim, out_channels=output_nc)
    else:
        print("Generator not implemented")
        exit()
    netG.apply(weights_init)
    netG.to(DEVICE)
    return netG


##############################################################################
# Generators
##############################################################################

## Pix2Pix Generator

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, use_skip=True):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)
        self.use_skip = use_skip

    def forward(self, x, skip_input=None):
        x = self.model(x)
        if self.use_skip and skip_input is not None:
            x = torch.cat((x, skip_input), 1)
        return x

class Pix2PixGenerator(nn.Module):
    def __init__(self, in_channels=801, out_channels=3):
        super(Pix2PixGenerator, self).__init__()

        # Spectral processing while maintaining 20x20 spatial dimension
        self.spectral_processing = nn.Sequential(
            nn.Conv2d(in_channels, 801, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(801, 512, kernel_size=1),
            nn.LeakyReLU(0.2)
        )

        # Reduce from 20x20 to 16x16
        self.initial_conv = nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=0)

        # Downsampling: 16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1
        self.down1 = UNetDown(512, 512, normalize=False)
        self.down2 = UNetDown(512, 512, dropout=0.5)
        self.down3 = UNetDown(512, 512, dropout=0.5)
        self.down4 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(0.2)
        )

        # Upsampling: 1x1 -> 2x2 -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(256, 128)
        self.up7 = UNetUp(128, 64)

        # Final upsampling to 256x256
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # x is 20x20x801
        x = self.spectral_processing(x)  # 20x20x512
        x = self.initial_conv(x)  # 16x16x512

        d1 = self.down1(x)  # 8x8x512
        d2 = self.down2(d1)  # 4x4x512
        d3 = self.down3(d2)  # 2x2x512
        d4 = self.down4(d3)  # 1x1x512

        bottleneck = self.bottleneck(d4)  # 1x1x512

        u1 = self.up1(bottleneck, d3)  # 2x2x(512+512)
        u2 = self.up2(u1, d2)  # 4x4x(512+512)
        u3 = self.up3(u2, d1)  # 8x8x(512+512)
        u4 = self.up4(u3, x)  # 16x16x(512+512)
        u5 = self.up5(u4)  # 32x32x256
        u6 = self.up6(u5)  # 64x64x128
        u7 = self.up7(u6)  # 128x128x64

        return self.final(u7)  # 256x256x3


###############

class ResidualGenerator(nn.Module):

    def __init__(self, input_dim, output_nc, ngf=64, n_downsampling=3, n_blocks=4, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResidualGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ConvTranspose2d(input_dim, ngf, 4, 1, 0, bias=False), norm_layer(ngf), activation]
        # model += [nn.Conv2d(middle_dimensions, ngf, kernel_size=3, stride=1, padding=1), norm_layer(ngf), activation]

        # upsample
        mult = 1
        n_upsampling=3
        for i in range(n_upsampling):
            model += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1), norm_layer(int(ngf * mult / 2)), activation]
            mult = mult/2

        for i in range(n_blocks):
            model += [ResnetBlock(int(ngf * mult), padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        n_upsampling = 3
        for i in range(n_upsampling):
            model += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult)), activation]

        model += [nn.Conv2d(int(ngf * mult), output_nc, kernel_size=3, padding=1)]

        # model = nn.Sequential(*model)
        # model.cuda()
        # summary(model, (38, 1, 1))
        # exit()

        self.model = nn.Sequential(*model)

    def forward(self, input):
        # prediction = self.linear(input)
        prediction = self.model(input)
        return prediction


def DCGan(input_dim):

    # define the standalone generator model
    layers = []
    ngf = 4

    layers.append(nn.ConvTranspose2d(input_dim, ngf * 32, 4, 1, 0, bias=False))
    layers.append(nn.BatchNorm2d(ngf * 32))
    layers.append(nn.ReLU(True))

    layers.append(nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False))
    layers.append(nn.BatchNorm2d(ngf * 16))
    layers.append(nn.ReLU(True))

    layers.append(nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False))
    layers.append(nn.BatchNorm2d(ngf * 8))
    layers.append(nn.ReLU(True))

    #32X32
    layers.append(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False))
    layers.append(nn.BatchNorm2d(ngf * 4))
    layers.append(nn.ReLU(True))

    #64X64
    out_channels = 1
    layers.append(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False))
    layers.append(nn.BatchNorm2d(ngf * 2))
    layers.append(nn.ReLU(True))

    #128X128
    out_channels = 1
    layers.append(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False))
    layers.append(nn.BatchNorm2d(ngf))
    layers.append(nn.ReLU(True))

    #256X256
    out_channels = 3
    layers.append(nn.ConvTranspose2d(ngf, out_channels, 4, 2, 1, bias=False))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=nn.Conv2d, which_bn=biggan_layers.bn, activation=None,
                 upsample=None, channel_ratio=4):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.in_channels // channel_ratio
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels,
                                     kernel_size=1, padding=0)
        self.conv2 = self.which_conv(self.hidden_channels, self.hidden_channels)
        self.conv3 = self.which_conv(self.hidden_channels, self.hidden_channels)
        self.conv4 = self.which_conv(self.hidden_channels, self.out_channels,
                                     kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(self.in_channels)
        self.bn2 = self.which_bn(self.hidden_channels)
        self.bn3 = self.which_bn(self.hidden_channels)
        self.bn4 = self.which_bn(self.hidden_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x, y):
        # Project down to channel ratio
        h = self.conv1(self.activation(self.bn1(x, y)))
        # Apply next BN-ReLU
        h = self.activation(self.bn2(h, y))
        # Drop channels in x if necessary
        if self.in_channels != self.out_channels:
            x = x[:, :self.out_channels]
            # Upsample both h and x at this point
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        # 3x3 convs
        h = self.conv2(h)
        h = self.conv3(self.activation(self.bn3(h, y)))
        # Final 1x1 conv
        h = self.conv4(self.activation(self.bn4(h, y)))
        return h + x


class BigGanGenerator(nn.Module):
    def __init__(self, G_ch=64, G_depth=2, dim_z=128, bottom_width=4, resolution=128,
                 G_kernel_size=3, G_attn='64', n_classes=1000,
                 num_G_SVs=1, num_G_SV_itrs=1,
                 G_shared=True, shared_dim=0, hier=False,
                 cross_replica=False, mybn=False,
                 G_activation=nn.ReLU(inplace=False),
                 G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
                 BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
                 G_init='ortho', skip_init=False, no_optim=False,
                 G_param='SN', norm_style='bn',
                 **kwargs):
        super(BigGanGenerator, self).__init__()
        # Channel width mulitplier
        self.ch = G_ch
        # Number of resblocks per stage
        self.G_depth = G_depth
        # Dimensionality of the latent space
        self.dim_z = dim_z
        # The initial spatial dimensions
        self.bottom_width = bottom_width
        # Resolution of the output
        self.resolution = resolution
        # Kernel size?
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # number of classes, for use in categorical conditional generation
        self.n_classes = n_classes
        # Use shared embeddings?
        self.G_shared = G_shared
        # Dimensionality of the shared embedding? Unused if not using G_shared
        # self.shared_dim = shared_dim if shared_dim > 0 else dim_z
        self.shared_dim = 0 #Srijay Changed Here
        # Hierarchical latent space?
        self.hier = hier
        # Cross replica batchnorm?
        self.cross_replica = cross_replica
        # Use my batchnorm?
        self.mybn = mybn
        # nonlinearity for residual blocks
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Normalization style
        self.norm_style = norm_style
        # Epsilon for BatchNorm?
        self.BN_eps = BN_eps
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # fp16?
        self.fp16 = G_fp16
        # Architecture dict
        ch = 64
        attention = '64'
        self.arch = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
               'upsample' : [True] * 6,
               'resolution' : [8, 16, 32, 64, 128, 256],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,9)}}

        # Which convs, batchnorms, and linear layers to use
        if self.G_param == 'SN':
            self.which_conv = functools.partial(biggan_layers.SNConv2d,
                                                kernel_size=3, padding=1,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                eps=self.SN_eps)
            self.which_linear = functools.partial(biggan_layers.SNLinear,
                                                  num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                  eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear

        # We use a non-spectral-normed embedding here regardless;
        # For some reason applying SN to G's embedding seems to randomly cripple G
        self.which_embedding = nn.Embedding
        bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                     else self.which_embedding)
        self.which_bn = functools.partial(biggan_layers.ccbn,
                                          which_linear=bn_linear,
                                          cross_replica=self.cross_replica,
                                          mybn=self.mybn,
                                          input_size=(self.shared_dim + 801 if self.G_shared
                                                      else self.n_classes),
                                          norm_style=self.norm_style,
                                          eps=self.BN_eps)

        # Prepare model
        # If not using shared embeddings, self.shared is just a passthrough
        self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared
                       else biggan_layers.identity())

        # First linear layer
        self.linear = self.which_linear(self.dim_z + self.shared_dim,
                                        self.arch['in_channels'][0] * (self.bottom_width ** 2))

        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        # while the inner loop is over a given block
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[GBlock(in_channels=self.arch['in_channels'][index],
                                    out_channels=self.arch['in_channels'][index] if g_index == 0 else
                                    self.arch['out_channels'][index],
                                    which_conv=self.which_conv,
                                    which_bn=self.which_bn,
                                    activation=self.activation,
                                    upsample=(functools.partial(F.interpolate, scale_factor=2)
                                              if self.arch['upsample'][index] and g_index == (
                                    self.G_depth - 1) else None))]
                            for g_index in range(self.G_depth)]

            # If attention on this block, attach it to the end
            if self.arch['attention'][self.arch['resolution'][index]]:
                print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
                self.blocks[-1] += [biggan_layers.Attention(self.arch['out_channels'][index], self.which_conv)]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        # output layer: batchnorm-relu-conv.
        # Consider using a non-spectral conv here
        self.output_layer = nn.Sequential(biggan_layers.bn(self.arch['out_channels'][-1],
                                                    cross_replica=self.cross_replica,
                                                    mybn=self.mybn),
                                          self.activation,
                                          self.which_conv(self.arch['out_channels'][-1], 3))

        # Initialize weights. Optionally skip init for testing.
        if not skip_init:
            self.init_weights()

        # Set up optimizer
        # If this is an EMA copy, no need for an optim, so just return now
        if no_optim:
            return
        self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
        if G_mixed_precision:
            print('Using fp16 adam in G...')
            import utils
            self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                                      betas=(self.B1, self.B2), weight_decay=0,
                                      eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0,
                                    eps=self.adam_eps)

            # LR scheduling, left here for forward compatibility
            # self.lr_sched = {'itr' : 0}# if self.progressive else {}
            # self.j = 0

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)

    # Note on this forward function: we pass in a y vector which has
    # already been passed through G.shared to enable easy class-wise
    # interpolation later. If we passed in the one-hot and then ran it through
    # G.shared in this forward function, it would be harder to handle.
    # NOTE: The z vs y dichotomy here is for compatibility with not-y
    def forward(self, y):
        z = torch.randn(y.shape[0], self.dim_z, requires_grad=False).to(DEVICE)
        # y = y.permute(0,2,3,1)

        # If hierarchical, concatenate zs and ys
        if self.hier:
            z = torch.cat([y, z], 1)
            y = z

        # First linear layer
        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h, y)

        output = torch.tanh(self.output_layer(h))
        return output
    