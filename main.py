import argparse
import math
from collections import defaultdict
import json
import csv
from torchvision.utils import save_image
import configparser
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from scipy.stats import pearsonr,spearmanr,kendalltau
from sklearn.metrics import r2_score
import torchvision.utils as vutils
from data.data import IRDataset
from model import GenerativeModel
from discriminators import Pix2PixDiscriminator, PatchDiscriminator, BigGanDiscriminator
from generators import weights_init
from losses import *
from utils import *
import os
import gradio as gr
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PIL import Image
from combat.pycombat import pycombat


parser = argparse.ArgumentParser()

parser.add_argument('--image_dir', default=r'./data/patches')

# Optimization hyperparameters
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_iterations', default=800000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)

# Switch the generator to eval mode after this many iterations
parser.add_argument('--eval_mode_after', default=9000000, type=int)

# Dataset options
parser.add_argument('--num_train_samples', default=10, type=int)
parser.add_argument('--num_val_samples', default=10, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=0, type=int)

# Image Generator options
#parser.add_argument('--generator', default='biggan')  # dcgan or pix2pix or residual or biggan
parser.add_argument('--generator', default='pix2pix')
parser.add_argument('--l1_pixel_image_loss_weight', default=100.0, type=float)  # 1.0
parser.add_argument('--normalization', default='instance')
parser.add_argument('--activation', default='leakyrelu-0.2')

# Generic discriminator options
parser.add_argument('--discriminator_loss_weight', default=1, type=float)  # 0.01
parser.add_argument('--gan_loss_type', default='gan')

# Image discriminator
#parser.add_argument('--discriminator', default='biggan')  # patchgan or standard or biggan
parser.add_argument('--discriminator', default='patchgan')

# Output options
parser.add_argument('--print_every', default=100, type=int)
parser.add_argument('--timing', default=False, type=bool_flag)
parser.add_argument('--checkpoint_every', default=1000, type=int)

# Experiment related parameters
#parser.add_argument('--experimentname', default='ir_spectra_generation_fromsinglevector_biggan_3_D')
parser.add_argument('--experimentname', default='ir_spectra_generation_pix2pix_D')
parser.add_argument('--output_dir', default=os.path.join('./output'))
parser.add_argument('--checkpoint_name', default='model.pt')

#parser.add_argument('--checkpoint_path', default='./trained_models/ir_spectra_generation_fromsinglevector_biggan_3_D/model/model.pt')
parser.add_argument('--checkpoint_path', default='./output/ir_spectra_generation_pix2pix_D/model/model.pt')
parser.add_argument('--restore_from_checkpoint', default=False, type=bool_flag)
parser.add_argument('--test_output_dir', default=os.path.join(r'./output'))

# If you want to test model, set mode to test, or gradio
parser.add_argument('--mode', default='train', type=str)


# Only used if using Wasserstein loss (not currently)
def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(real_samples.DEVICE)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    
    fake = torch.ones_like(d_interpolates).to(real_samples.DEVICE)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    curr_loss = curr_loss * weight
    loss_dict[loss_name] = curr_loss.item() if torch.is_tensor(curr_loss) else curr_loss
    if total_loss is not None:
        if torch.is_tensor(total_loss) and torch.is_tensor(curr_loss):
            total_loss += curr_loss
        else:
            total_loss += torch.tensor(curr_loss, device=total_loss.device) if torch.is_tensor(total_loss) else curr_loss
    else:
        total_loss = curr_loss
    return total_loss


def build_dsets(args):
    dset_kwargs = {
        'base_dir': args.image_dir,
        'mode': args.mode
    }

    dset = IRDataset(**dset_kwargs)

    num_imgs = len(dset)
    print(args.mode + ' dataset has %d images' % (num_imgs))

    return dset


def build_loader(args):
    dset = build_dsets(args)

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': True,
    }

    loader = DataLoader(dset, **loader_kwargs)

    return loader


def build_model(args):
    kwargs = {
        'normalization': args.normalization,
        'activation': args.activation,
        'mode': args.mode,
        'generator_name': args.generator
    }
    model = GenerativeModel(**kwargs)
    return model, kwargs


def build_img_discriminator(args):
    if (args.discriminator == 'patchgan'):
        discriminator = Pix2PixDiscriminator(in_channels=3)
    elif (args.discriminator == 'standard'):
        d_kwargs = {
            'arch': args.d_img_arch,
            'normalization': args.d_normalization,
            'activation': args.d_activation,
            'padding': args.d_padding,
        }
        discriminator = PatchDiscriminator(**d_kwargs)
    elif (args.discriminator == 'biggan'):
        discriminator = BigGanDiscriminator()
    else:
        raise 'Give proper name of discriminator'

    discriminator = discriminator.apply(weights_init)

    return discriminator


def check_model(args, t, loader, model, mode):
    experiment_output_dir = os.path.join(args.output_dir, args.experimentname)
    output_dir = os.path.join(experiment_output_dir, 'training_output', mode)
    mkdir(output_dir)
    # model.eval()
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            patch_id, ir_features, image_gt = batch

            ir_features = ir_features.to(dtype=torch.float32, device=DEVICE)
            image_gt = image_gt.to(dtype=torch.float32, device=DEVICE)

            image_pred = model(ir_features=ir_features.float())
            image_pred = image_pred.to(DEVICE)

            n_samples += 1
            if n_samples >= 5:
                break

            ir_features_np = ir_features.cpu().numpy()
            batch_size = ir_features_np.shape[0]

            # Text file to store IR feature stats
            stats_file_path = os.path.join(output_dir, f'ir_features_stats_batch_{n_samples}.txt')
            with open(stats_file_path, 'w') as f:
                for i in range(batch_size):
                    patch_features = ir_features_np[i]
                    f.write(f'Patch {patch_id[i]} IR Features Statistics:\n')
                    f.write(f'  Shape: {patch_features.shape}\n')
                    f.write(f'  Min: {np.min(patch_features)}\n')
                    f.write(f'  Max: {np.max(patch_features)}\n')
                    f.write(f'  Mean: {np.mean(patch_features)}\n')
                    f.write(f'  Std: {np.std(patch_features)}\n')
                    f.write(f'  Median: {np.median(patch_features)}\n')
                    f.write(f'  Non-zero elements: {np.count_nonzero(patch_features)}\n')
                    f.write('\n')

            image_gt_path = os.path.join(output_dir, patch_id[0] + '_gt_image.png')
            save_image(image_gt, image_gt_path)

            if (image_pred is not None):
                image_pred_path = os.path.join(output_dir, patch_id[0] + '_pred_image.png')
                save_image(image_pred, image_pred_path)


def compute_metrics(total_real_spectra, total_predicted_spectra):
    #Compute pearson, spearman, kendalltau and r2
    metric_dict = {}
    num_wavenumbers = total_real_spectra.shape[1]

    for i in range(num_wavenumbers):
        real_intensities = total_real_spectra[:, i]
        predicted_intensities = total_predicted_spectra[:, i]
        metric_dict[f'wavenumber_{i}'] = [
            pearsonr(real_intensities, predicted_intensities)[0],
            spearmanr(real_intensities, predicted_intensities)[0],
            kendalltau(real_intensities, predicted_intensities)[0],
            r2_score(real_intensities, predicted_intensities)
        ]
    
    # Compute overall metrics
    metric_dict['overall'] = [
        pearsonr(total_real_spectra.flatten(), total_predicted_spectra.flatten())[0],
        spearmanr(total_real_spectra.flatten(), total_predicted_spectra.flatten())[0],
        kendalltau(total_real_spectra.flatten(), total_predicted_spectra.flatten())[0],
        r2_score(total_real_spectra.flatten(), total_predicted_spectra.flatten())
    ]
    
    return metric_dict


def generate_report(report_name, metric_dict):
    output_file = os.path.join(args.test_output_dir, report_name + '.csv')
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['wavenumber', 'pearson', 'spearman', 'kendalltau', 'r2'])
        for wavenumber, metrics in metric_dict.items():
            writer.writerow([wavenumber] + [str(metric) for metric in metrics])

    # Generate summary stats
    summary_file = os.path.join(args.test_output_dir, report_name + '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write('Summary Statistics:\n')
        for stat in ['pearson', 'spearman', 'kendalltau', 'r2']:
            values = [metrics[['pearson', 'spearman', 'kendalltau', 'r2'].index(stat)] 
                      for metrics in metric_dict.values() if wavenumber != 'overall']
            f.write(f'{stat.capitalize()}:\n')
            f.write(f'  Mean: {np.mean(values):.4f}\n')
            f.write(f'  Median: {np.median(values):.4f}\n')
            f.write(f'  Std Dev: {np.std(values):.4f}\n')
            f.write(f'  Min: {np.min(values):.4f}\n')
            f.write(f'  Max: {np.max(values):.4f}\n\n')

def test_model(args, loader, model, image_discriminator):
    gt_image_output_dir = os.path.join(args.test_output_dir,'real')
    pred_image_output_dir = os.path.join(args.test_output_dir,'synthetic')
    mkdir(gt_image_output_dir)
    mkdir(pred_image_output_dir)

    real_spectra = []
    pred_spectra = []

    with torch.no_grad():

        for batch in loader:
            patch_id, ir_spectra, image_gt = batch

            ir_spectra = ir_spectra.to(dtype=torch.float32, device=DEVICE)
            image_gt = image_gt.to(dtype=torch.float32, device=DEVICE)

            image_pred = model(ir_features=ir_spectra.float())
            image_pred = image_pred.to(DEVICE)

            if args.discriminator == 'biggan':
                _, predicted_spectra = image_discriminator(image_gt.float(), ir_spectra.float())
                
                real_spectra_batch_np = ir_spectra.cpu().detach().numpy().tolist()
                pred_spectra_batch_np = predicted_spectra.cpu().detach().numpy().tolist()

                real_spectra = real_spectra + real_spectra_batch_np
                pred_spectra = pred_spectra + pred_spectra_batch_np

            image_gt_path = os.path.join(gt_image_output_dir, f'{patch_id[0]}.png')
            save_image(image_gt, image_gt_path)

            image_pred_path = os.path.join(pred_image_output_dir, f'{patch_id[0]}.png')
            save_image(image_pred, image_pred_path)

        if args.discriminator == 'biggan':
            real_spectra = np.array(real_spectra)
            pred_spectra = np.array(pred_spectra)
            np.save('D_real.py', real_spectra)
            np.save('D_syn.py', pred_spectra)

            # Generate overall report
            overall_metrics = compute_metrics(real_spectra, pred_spectra)
            generate_report(args.experimentname + '_overall_report', overall_metrics)


def calculate_model_losses(args, image_gt, image_pred):
    total_loss = torch.tensor(0.0, device=image_gt.device)
    losses = {}

    # Image L1 Loss
    l1_pixel_loss_images = F.l1_loss(image_pred, image_gt.float())
    total_loss = add_loss(total_loss, l1_pixel_loss_images, losses, 'L1_pixel_loss_images',
                          args.l1_pixel_image_loss_weight)
    
    # Added structure consistency loss for testing
    structure_loss = structure_consistency_loss(image_pred, image_gt.float())
    total_loss = add_loss(total_loss, structure_loss, losses, 'structure_consistency_loss',
                          args.l1_pixel_image_loss_weight)

    # Prediction losses if required (BigGAN performs generation + prediction; Pix2Pix GAN only performs generation currently)

    #ir_prediction_loss = F.mse_loss(ir_pred, ir_gt.float())
    #total_loss = add_loss(total_loss, ir_prediction_loss, losses, 'IR_prediction_loss',
    #                      args.ir_prediction_loss_weight)

    # l2_prediction_loss = F.mse_loss(pred_expression, gt_expression.float())
    # total_loss = add_loss(total_loss, l2_prediction_loss, losses, 'prediction_loss',
    #                       args.l1_pixel_image_loss_weight)

    return total_loss, losses


def main(args):
    torch.cuda.empty_cache() if DEVICE == 'cuda' else torch.mps.empty_cache() if DEVICE == 'mps' else None

    experiment_output_dir = os.path.join(args.output_dir, args.experimentname)
    model_dir = os.path.join(experiment_output_dir, 'model')

    float_dtype = torch.float32

    if (args.mode == 'train'):
        mkdir(experiment_output_dir)
        mkdir(model_dir)

        with open(os.path.join(experiment_output_dir, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    loader = build_loader(args)

    model, model_kwargs = build_model(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5,0.999))
    
    # For Wasserstein loss:
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=0.99)

    # Image Discriminator
    image_discriminator = build_img_discriminator(args)
    if image_discriminator is not None:
        image_discriminator.to(DEVICE)
        image_discriminator.type(float_dtype)
        image_discriminator.train()
        optimizer_d_image = torch.optim.Adam(image_discriminator.parameters(), lr=args.learning_rate, betas=(0.5,0.999))

        # For Wasserstein loss:
        #optimizer_d_image = torch.optim.RMSprop(image_discriminator.parameters(), lr=args.learning_rate, alpha=0.99)

    gan_g_loss, gan_d_loss = get_gan_losses(args.gan_loss_type)

    if args.restore_from_checkpoint or args.mode == 'test' or args.mode == 'gradio' or args.mode == 'interpolate':

        print('Restoring')
        restore_path = args.checkpoint_path

        checkpoint = torch.load(restore_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state'], strict=True)

        if (args.mode == 'train'):
            optimizer.load_state_dict(checkpoint['optim_state']) #strict argument is not supported here

        if image_discriminator is not None:
            image_discriminator.load_state_dict(checkpoint['d_image_state'])
            optimizer_d_image.load_state_dict(checkpoint['d_image_optim_state'])
            image_discriminator.to(DEVICE)

        if (args.mode == 'test'):
            model.eval()
            test_model(args, loader, model, image_discriminator)
            print('Testing has been done and results are saved')
            return

        t = 0
        epoch = checkpoint['counters']['epoch']
        print('Starting Epoch : ', epoch)

    else:

        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'model_kwargs': model_kwargs,
            'losses_ts': [],
            'losses': defaultdict(list),
            'd_losses': defaultdict(list),
            'checkpoint_ts': [],
            'train_batch_data': [],
            'train_samples': [],
            'train_iou': [],
            'val_batch_data': [],
            'val_samples': [],
            'val_losses': defaultdict(list),
            'val_iou': [],
            'norm_d': [],
            'norm_g': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'model_state': None, 'model_best_state': None, 'optim_state': None,
            'd_obj_state': None, 'd_obj_best_state': None, 'd_obj_optim_state': None,
            'd_img_state': None, 'd_img_best_state': None, 'd_img_optim_state': None,
            'd_mask_state': None, 'best_t': [],
        }

    # Loss Curves
    training_loss_out_dir = os.path.join(experiment_output_dir, 'training_loss_graph')
    mkdir(training_loss_out_dir)

    def draw_curve(epoch_list, loss_list, loss_name):
        plt.clf()
        plt.plot(epoch_list, loss_list, 'bo-', label=loss_name)
        plt.legend()
        plt.savefig(os.path.join(training_loss_out_dir, loss_name + '.png'))

    epoch_list = []
    monitor_epoch_losses = defaultdict(list)

    while True:

        if t >= args.num_iterations:
            break

        for batch in loader:

            if t == args.eval_mode_after:
                print('switching to eval mode')
                model.eval()
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            
            patch_id, ir_spectra, image_gt = batch

            ir_spectra = ir_spectra.to(dtype=torch.float32, device=DEVICE)
            image_gt = image_gt.to(dtype=torch.float32, device=DEVICE)

            image_pred = model(ir_features=ir_spectra.float())
            image_pred = image_pred.to(DEVICE)

            total_loss, losses = calculate_model_losses(args, image_gt, image_pred)

            ## Generator Training
            if image_discriminator is not None:
                if(args.discriminator == 'biggan'):
                    scores_image_fake, fake_prediction = image_discriminator(image_pred, ir_spectra.float())
                    scores_image_fake = scores_image_fake.to(DEVICE)
                    weight = args.discriminator_loss_weight
                    total_loss = add_loss(total_loss, gan_g_loss(scores_image_fake), losses, 'g_gan_image_loss', weight)

                    l2_prediction_loss = F.mse_loss(fake_prediction, ir_spectra.float())
                    total_loss = add_loss(total_loss, l2_prediction_loss, losses, 'fake_g_prediction_loss',
                                        args.l1_pixel_image_loss_weight)
                else: # patchgan
                    scores_image_fake = image_discriminator(image_pred)
                    scores_image_fake = scores_image_fake.to(DEVICE)
                    weight = args.discriminator_loss_weight
                    total_loss = add_loss(total_loss, gan_g_loss(scores_image_fake), losses, 'g_gan_image_loss', weight)

            losses['total_loss'] = total_loss.item()
            if not math.isfinite(losses['total_loss']):
                print('WARNING: Got loss = NaN, not backpropping')
                continue

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            image_fake = image_pred.detach()
            image_real = image_gt.detach()

            ## Discriminator Training
            if image_discriminator is not None:
                d_image_losses = LossManager()  # For image
                if (args.discriminator == 'biggan'):
                    scores_fake, _ = image_discriminator(image_fake, ir_spectra.float())
                    scores_real, real_d_prediction = image_discriminator(image_real.float(), ir_spectra.float())

                    d_image_gan_loss = gan_d_loss(scores_real, scores_fake)
                    d_image_losses.add_loss(d_image_gan_loss, 'd_image_gan_loss')

                    l2_prediction_loss = F.mse_loss(real_d_prediction, ir_spectra.float())
                    d_image_losses.add_loss(l2_prediction_loss, 'real_d_prediction_loss')
                else:
                    scores_fake = image_discriminator(image_fake)
                    scores_real = image_discriminator(image_real.float())

                    d_image_gan_loss = gan_d_loss(scores_real, scores_fake)
                    d_image_losses.add_loss(d_image_gan_loss, 'd_image_gan_loss')

                optimizer_d_image.zero_grad()
                d_image_losses.total_loss.backward()
                optimizer_d_image.step()
                image_discriminator.to(DEVICE)

            t += 1

            ## Output progress / save checkpoint
            if t % args.print_every == 0:

                print('t = %d / %d' % (t, args.num_iterations))
                for name, val in losses.items():
                    print(' G [%s]: %.4f' % (name, val))

                if image_discriminator is not None:
                    for name, val in d_image_losses.items():
                        print(' D_img [%s]: %.4f' % (name, val))

            if t % args.checkpoint_every == 0:

                print('checking on train')
                check_model(args, t, loader, model, 'train')

                checkpoint['model_state'] = model.state_dict()

                if image_discriminator is not None:
                    checkpoint['d_image_state'] = image_discriminator.state_dict()
                    checkpoint['d_image_optim_state'] = optimizer_d_image.state_dict()

                checkpoint['optim_state'] = optimizer.state_dict()
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint_path = os.path.join(model_dir, args.checkpoint_name)
                print('Saving checkpoint to ', checkpoint_path)
                torch.save(checkpoint, checkpoint_path)

        # Plot the loss curves
        epoch += 1
        epoch_list.append(epoch)
        for k, v in losses.items():
            monitor_epoch_losses[k].append(v)
            draw_curve(epoch_list, monitor_epoch_losses[k], k)


if __name__ == '__main__':
    print('CONTROL')
    args = parser.parse_args()
    main(args)
