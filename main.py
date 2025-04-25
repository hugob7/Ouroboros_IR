import argparse
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim

from collections import defaultdict
from scipy import stats
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data.data import IRDataset
from discriminators import Pix2PixDiscriminator, PatchDiscriminator, BigGanDiscriminator
from generators import weights_init
from losses import get_gan_losses
from metrics import calculate_fid_score, save_spectra_comparison
from model import GenerativeModel
from utils import *


parser = argparse.ArgumentParser()

parser.add_argument("--image_dir", default='./data/patches')

# Optimization hyperparameters
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--num_iterations", default=800000, type=int)
parser.add_argument("--learning_rate", default=1e-4, type=float)

# Switch the generator to eval mode after this many iterations
parser.add_argument("--eval_mode_after", default=9000000, type=int)

# Dataset options
parser.add_argument("--num_train_samples", default=10, type=int)
parser.add_argument("--num_val_samples", default=10, type=int)
parser.add_argument("--shuffle_val", default=True, type=bool_flag)
parser.add_argument("--loader_num_workers", default=0, type=int)

# Image Generator options
parser.add_argument("--generator", default="biggan")  # Options: dcgan | pix2pix | residual | biggan
parser.add_argument("--l1_pixel_image_loss_weight", default=100.0, type=float)  # 1.0
parser.add_argument("--normalization", default="instance")
parser.add_argument("--activation", default="leakyrelu-0.2")

# Generic discriminator options
parser.add_argument("--discriminator_loss_weight", default=1, type=float)  # 0.01
parser.add_argument("--gan_loss_type", default="gan")

# Image discriminator
parser.add_argument("--discriminator", default="biggan")  # Options: patchgan | standard | biggan

# Output options
parser.add_argument("--print_every", default=100, type=int)
parser.add_argument("--timing", default=False, type=bool_flag)
parser.add_argument("--checkpoint_every", default=300, type=int)

# Experiment related parameters
parser.add_argument("--experimentname", default="ir_model_name")
parser.add_argument("--output_dir", default=os.path.join("./output"))
parser.add_argument("--checkpoint_name", default="model.pt")
parser.add_argument("--checkpoint_path", default="./output/ir_model_name/model/model.pt")
parser.add_argument("--restore_from_checkpoint", default=True, type=bool_flag)
parser.add_argument("--test_output_dir", default=os.path.join("./output"))

# If you want to test model, set mode to test, or gradio
parser.add_argument("--mode", default="train", type=str)


def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    curr_loss = curr_loss * weight
    loss_dict[loss_name] = curr_loss.item()
    if total_loss is not None:
        total_loss += curr_loss
    else:
        total_loss = curr_loss
    return total_loss

def build_dsets(args, mode=None):
    curr_mode = mode if mode is not None else args.mode

    dset_kwargs = {
        "base_dir": args.image_dir,
        "mode": curr_mode
    }

    dset = IRDataset(**dset_kwargs)
    print(f"{curr_mode.capitalize()} dataset has {len(dset)} images")
    return dset

def build_loader(args, mode=None):
    dset = build_dsets(args, mode=mode)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.loader_num_workers,
        "shuffle": True,
    }

    loader = DataLoader(dset, **loader_kwargs)
    return loader

def build_model(args):
    kwargs = {
        "normalization": args.normalization,
        "activation": args.activation,
        "mode": args.mode,
        "generator_name": args.generator
    }
    model = GenerativeModel(**kwargs)
    return model, kwargs

def build_img_discriminator(args):
    if args.discriminator == "patchgan":
        discriminator = Pix2PixDiscriminator()
    elif args.discriminator == "standard":
        d_kwargs = {
            "arch": args.d_img_arch,
            "normalization": args.d_normalization,
            "activation": args.d_activation,
            "padding": args.d_padding,
        }
        discriminator = PatchDiscriminator(**d_kwargs)
    elif args.discriminator == "biggan":
        discriminator = BigGanDiscriminator()
    else:
        raise "Give proper name of discriminator"

    discriminator = discriminator.apply(weights_init)
    return discriminator


def check_model(args, epoch, loader, model, discriminator, mode):
    experiment_output_dir = os.path.join(args.output_dir, args.experimentname)
    output_dir = os.path.join(experiment_output_dir, "training_output", mode)
    mkdir(output_dir)

    model.eval()
    discriminator.eval()

    real_images = []
    generated_images = []
    prediction_mses = []
    pearson_correlations = []
    spearman_correlations = []

    batches_to_save = 5

    print(f"Evaluating model on {mode} set (epoch {epoch})...")

    with torch.no_grad():
        for batch_idx, (_, ir_features, image_gt) in enumerate(loader):
            ir_features = ir_features.to(dtype=torch.float32, device=DEVICE)
            image_gt = image_gt.to(dtype=torch.float32, device=DEVICE)

            ir_vector_target = torch.mean(ir_features, dim=(2, 3)) # 801d target (gt) spectra vector

            # Generate synthetic H&E image
            image_pred = model(ir_features=ir_features.float())
            image_pred = image_pred.to(DEVICE)

            real_images.append(image_gt.cpu().numpy())
            generated_images.append(image_pred.cpu().numpy())

            # Predict 801d averaged spectra vector
            _, ir_prediction = discriminator(image_gt)
            
            real_mse = F.mse_loss(ir_prediction, ir_vector_target).item()
            prediction_mses.append(real_mse)

            # Calculate correlations for all samples in batch
            for i in range(ir_vector_target.size(0)):
                target = ir_vector_target[i].cpu().numpy()
                pred = ir_prediction[i].cpu().numpy()
                
                # Pearson and Spearman correlation
                pearson_corr, _ = stats.pearsonr(target, pred)
                pearson_correlations.append(pearson_corr)
                spearman_corr, _ = stats.spearmanr(target, pred)
                spearman_correlations.append(spearman_corr)
            
            # Save example spectra predictions vs ground truth
            if batch_idx < batches_to_save:
                target_0 = ir_vector_target[0].cpu().numpy()
                pred_0 = ir_prediction[0].cpu().numpy()
                pearson_0, _ = stats.pearsonr(target_0, pred_0)
                spearman_0, _ = stats.spearmanr(target_0, pred_0)

                fpath = os.path.join(output_dir, f"epoch{epoch}_batch{batch_idx}_spectra.png")
                save_spectra_comparison(target_0, pred_0, fpath, pearson_0, spearman_0)

                image_gt_path = os.path.join(output_dir, f"epoch{epoch}_batch{batch_idx}_real.png")
                image_pred_path = os.path.join(output_dir, f"epoch{epoch}_batch{batch_idx}_pred.png")
                save_image(image_gt, image_gt_path)
                save_image(image_pred, image_pred_path)
    
    fid_score = calculate_fid_score(real_images, generated_images)
    avg_prediction_mse = np.mean(prediction_mses)
    avg_pearson = np.mean(pearson_correlations)
    avg_spearman = np.mean(spearman_correlations)
    
    print(f"FID Score: {fid_score}")
    print(f"Average Spectra Prediction MSE: {avg_prediction_mse:.6f}")
    print(f"Average Pearson Correlation: {avg_pearson:.6f}")
    print(f"Average Spearman Correlation: {avg_spearman:.6f}")
    
    with open(os.path.join(output_dir, f"metrics_epoch{epoch}.txt"), 'w') as f:
        f.write(f"FID Score: {fid_score}\n")
        f.write(f"Average Spectral Prediction MSE: {avg_prediction_mse:.6f}\n")
        f.write(f"Average Pearson Correlation: {avg_pearson:.6f}\n")
        f.write(f"Average Spearman Correlation: {avg_spearman:.6f}\n")


def test_model(args, loader, model, discriminator):
    gt_image_output_dir = os.path.join(args.test_output_dir, "real")
    pred_image_output_dir = os.path.join(args.test_output_dir, "synthetic")
    spectra_output_dir = os.path.join(args.test_output_dir, "spectra")
    metrics_output_dir = os.path.join(args.test_output_dir, "metrics")
    mkdir(gt_image_output_dir)
    mkdir(pred_image_output_dir)
    mkdir(spectra_output_dir)
    mkdir(metrics_output_dir)
    
    model.eval()
    discriminator.eval()
    
    real_images = []
    generated_images = []
    prediction_mses = []
    pearson_correlations = []
    spearman_correlations = []
    patch_ids = []
    
    print("Testing model...")
    
    with torch.no_grad():
        for batch_idx, (patch_id, ir_features, image_gt) in enumerate(loader):
            patch_ids.append(patch_id[0]) 
            
            ir_features = ir_features.to(dtype=torch.float32, device=DEVICE)
            image_gt = image_gt.to(dtype=torch.float32, device=DEVICE)
            
            ir_vector_target = torch.mean(ir_features, dim=(2, 3))  # 801d target spectra vector
            
            # Generate synthetic H&E image
            image_pred = model(ir_features=ir_features.float())
            image_pred = image_pred.to(DEVICE)
            
            real_images.append(image_gt.cpu().numpy())
            generated_images.append(image_pred.cpu().numpy())
            
            # Predict 801d spectra vector from ground truth image
            _, ir_prediction = discriminator(image_gt)
            
            spectra_mse = F.mse_loss(ir_prediction, ir_vector_target).item()
            prediction_mses.append(spectra_mse)
            
            # Calculate correlations
            for i in range(ir_vector_target.size(0)):  # In case batch size != 1 for testing
                target = ir_vector_target[i].cpu().numpy()
                pred = ir_prediction[i].cpu().numpy()
                
                # Pearson and Spearman correlation
                pearson_corr, _ = stats.pearsonr(target, pred)
                pearson_correlations.append(pearson_corr)
                spearman_corr, _ = stats.spearmanr(target, pred)
                spearman_correlations.append(spearman_corr)
                                
                fpath = os.path.join(spectra_output_dir, f"{patch_id[0]}_spectra.png")
                save_spectra_comparison(target, pred, fpath, pearson_corr, spearman_corr)
            
            # Save corresponding synthetic and real H&E images
            image_gt_path = os.path.join(gt_image_output_dir, f"{patch_id[0]}.png")
            image_pred_path = os.path.join(pred_image_output_dir, f"{patch_id[0]}.png")
            save_image(image_gt, image_gt_path)
            save_image(image_pred, image_pred_path)
        
        # Calculate overall (averaged) metrics
        fid_score = calculate_fid_score(real_images, generated_images)
        avg_prediction_mse = np.mean(prediction_mses)
        avg_pearson = np.mean(pearson_correlations)
        avg_spearman = np.mean(spearman_correlations)
        
        print("\nTest Results:")
        print(f"FID Score: {fid_score}")
        print(f"Average Spectra Prediction MSE: {avg_prediction_mse:.6f}")
        print(f"Average Pearson Correlation: {avg_pearson:.6f}")
        print(f"Average Spearman Correlation: {avg_spearman:.6f}")

        # Save results (per patch) to CSV file
        metrics_df = pd.DataFrame({"Patch_ID": patch_ids, "MSE": prediction_mses, "Pearson": pearson_correlations, "Spearman": spearman_correlations})
        metrics_df.to_csv(os.path.join(metrics_output_dir, "sample_metrics.csv"), index=False)

        with open(os.path.join(args.test_output_dir, "summary_metrics.txt"), 'w') as f:
            f.write(f"FID Score: {fid_score}\n")
            f.write(f"Average Spectral Prediction MSE: {avg_prediction_mse:.6f}\n")
            f.write(f"Average Pearson Correlation: {avg_pearson:.6f}\n")
            f.write(f"Average Spearman Correlation: {avg_spearman:.6f}\n")
        

def calculate_model_losses(args, image_gt, image_pred):
    total_loss = torch.tensor(0.0, device=image_gt.device)
    losses = {}

    # Image L1 Loss
    l1_pixel_loss_images = F.l1_loss(image_pred, image_gt.float())
    total_loss = add_loss(total_loss, l1_pixel_loss_images, losses, "L1_pixel_loss_images", args.l1_pixel_image_loss_weight)
    
    return total_loss, losses


def main(args):

    torch.cuda.empty_cache() if DEVICE == "cuda" else torch.mps.empty_cache() if DEVICE == "mps" else None

    experiment_output_dir = os.path.join(args.output_dir, args.experimentname)
    model_dir = os.path.join(experiment_output_dir, "model")

    if args.mode == "train":
        mkdir(experiment_output_dir)
        mkdir(model_dir)

        with open(os.path.join(experiment_output_dir, "config.txt"), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # Train or test loader depending on mode set by user
    loader = build_loader(args)

    # Model (Generator)
    model, model_kwargs = build_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5,0.999))

    # Discriminator
    image_discriminator = build_img_discriminator(args)
    image_discriminator.to(DEVICE)
    image_discriminator.type(torch.float32)
    image_discriminator.train()
    optimizer_d_image = torch.optim.Adam(image_discriminator.parameters(), lr=args.learning_rate, betas=(0.5,0.999))

    gan_g_loss, gan_d_loss = get_gan_losses(args.gan_loss_type)

    if args.restore_from_checkpoint or args.mode == "test":
        print("Restoring model...")

        checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint['model_state'], strict=True)

        if args.mode == "train":
            optimizer.load_state_dict(checkpoint["optim_state"]) #strict argument is not supported here

        image_discriminator.load_state_dict(checkpoint["d_image_state"])
        optimizer_d_image.load_state_dict(checkpoint["d_image_optim_state"])
        image_discriminator.to(DEVICE)

        if args.mode == "test":
            model.eval()
            test_model(args, loader, model, image_discriminator)
            print("Testing phase completed and results saved.")
            return

        t = 0

        epoch = checkpoint["counters"]["epoch"]
        print("Starting Epoch : ", epoch)

    else:

        t, epoch = 0, 0
        checkpoint = {
            "args": args.__dict__,
            "model_kwargs": model_kwargs,
            "losses_ts": [],
            "losses": defaultdict(list),
            "d_losses": defaultdict(list),
            "checkpoint_ts": [],
            "train_batch_data": [],
            "train_samples": [],
            "train_iou": [],
            "val_batch_data": [],
            "val_samples": [],
            "val_losses": defaultdict(list),
            "val_iou": [],
            "norm_d": [],
            "norm_g": [],
            "counters": {
                't': None,
                "epoch": None,
            },
            "model_state": None, "model_best_state": None, "optim_state": None,
            "d_obj_state": None, "d_obj_best_state": None, "d_obj_optim_state": None,
            "d_img_state": None, "d_img_best_state": None, "d_img_optim_state": None,
            "d_mask_state": None, "best_t": [],
        }

    # Loss curves
    training_loss_out_dir = os.path.join(experiment_output_dir, "training_loss_graph")
    mkdir(training_loss_out_dir)

    def draw_curve(epoch_list, loss_list, loss_name):
        plt.clf()
        plt.plot(epoch_list, loss_list, "bo-", label=loss_name)
        plt.legend()
        plt.savefig(os.path.join(training_loss_out_dir, loss_name + ".png"))

    epoch_list = []
    monitor_epoch_losses = defaultdict(list)

    # t measures number of batches processed (iterations of training loop)
    while True:

        if t >= args.num_iterations:
            break

        #print(f"Epoch: {epoch}")

        for _, ir_spectra, image_gt in loader:
            model.train()

            if t == args.eval_mode_after:
                print(f"{args.eval_mode_after} iterations completed. Switching model to eval mode...")
                model.eval()
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            
            ir_spectra = ir_spectra.to(dtype=torch.float32, device=DEVICE) # 20x20x801
            image_gt = image_gt.to(dtype=torch.float32, device=DEVICE) # 256x256x3

            ir_vector_target = torch.mean(ir_spectra, dim=(2, 3)) # 801d vector

            image_pred = model(ir_features=ir_spectra.float())
            image_pred = image_pred.to(DEVICE)

            total_loss, losses = calculate_model_losses(args, image_gt, image_pred)

            # Generator Training
            if args.discriminator == "biggan":
                scores_image_fake, fake_prediction = image_discriminator(image_pred)
                scores_image_fake = scores_image_fake.to(DEVICE)

                weight = args.discriminator_loss_weight
                total_loss = add_loss(total_loss, gan_g_loss(scores_image_fake), losses, "g_gan_image_loss", weight)

                l2_prediction_loss = F.mse_loss(fake_prediction, ir_vector_target)
                total_loss = add_loss(total_loss, l2_prediction_loss, losses, "fake_g_prediction_loss",
                                        args.l1_pixel_image_loss_weight)
            else:
                scores_image_fake = image_discriminator(image_pred, ir_spectra.float())
                scores_image_fake = scores_image_fake.to(DEVICE)
                weight = args.discriminator_loss_weight
                total_loss = add_loss(total_loss, gan_g_loss(scores_image_fake), losses, "g_gan_image_loss", weight)

            losses["total_loss"] = total_loss.item()
            if not math.isfinite(losses["total_loss"]):
                print("WARNING: Got loss = NaN, not backpropping")
                continue

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            image_fake = image_pred.detach()
            image_real = image_gt.detach()

            # Discriminator Training
            d_image_losses = LossManager()  # For image
            if args.discriminator == "biggan":
                scores_fake, _ = image_discriminator(image_fake)
                scores_real, real_d_prediction = image_discriminator(image_real.float())

                d_image_gan_loss = gan_d_loss(scores_real, scores_fake)
                d_image_losses.add_loss(d_image_gan_loss, "d_image_gan_loss")

                l2_prediction_loss = F.mse_loss(real_d_prediction, ir_vector_target)
                d_image_losses.add_loss(l2_prediction_loss, "real_d_prediction_loss")
            else:
                scores_fake = image_discriminator(image_fake, ir_spectra.float())
                scores_real = image_discriminator(image_real.float(), ir_spectra.float())

                d_image_gan_loss = gan_d_loss(scores_real, scores_fake)
                d_image_losses.add_loss(d_image_gan_loss, "d_image_gan_loss")

            optimizer_d_image.zero_grad()
            d_image_losses.total_loss.backward()
            optimizer_d_image.step()
            image_discriminator.to(DEVICE)

            t += 1

            # Output progress
            if t % args.print_every == 0:
                print(f"t = {t} / {args.num_iterations}")
                for name, val in losses.items():
                    print(f" G [{name}]: {val:.4f}")

                if image_discriminator is not None:
                    for name, val in d_image_losses.items():
                        print(f" D_img [{name}]: {val:.4f}")

            # Save checkpoint
            if t % args.checkpoint_every == 0:
                check_model(args, epoch, loader, model, image_discriminator, "train")

                checkpoint["model_state"] = model.state_dict()

                checkpoint["d_image_state"] = image_discriminator.state_dict()
                checkpoint["d_image_optim_state"] = optimizer_d_image.state_dict()

                checkpoint["optim_state"] = optimizer.state_dict()
                checkpoint["counters"]['t'] = t
                checkpoint["counters"]["epoch"] = epoch
                checkpoint_path = os.path.join(model_dir, args.checkpoint_name)
                print("Saving checkpoint to ", checkpoint_path)
                torch.save(checkpoint, checkpoint_path)

        # Plot loss curves per epoch
        epoch += 1
        epoch_list.append(epoch)
        for k, v in losses.items():
            monitor_epoch_losses[k].append(v)
            draw_curve(epoch_list, monitor_epoch_losses[k], k)


if __name__ == '__main__':
    print("CONTROL")
    args = parser.parse_args()
    main(args)
