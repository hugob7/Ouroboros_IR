import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import linalg
from torchvision.models import inception_v3
from utils import DEVICE


def inception_get_features(inception, images):
    features = []
    batch_size = 32

    for i in range(0, images.shape[0], batch_size):
        batch = images[i: i+batch_size].to(DEVICE)

        if batch.shape[2] != 299 or batch.shape[3] != 299:
            batch = torch.nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)

        with torch.no_grad():
            feat = inception(batch)
        
        features.append(feat.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features


def calculate_fid_score(real_images, fake_images):
    real_images, fake_images = np.concatenate(real_images, axis=0), np.concatenate(fake_images, axis=0)     
    real_images, fake_images = torch.from_numpy(real_images).float(), torch.from_numpy(fake_images).float()
    
    inception = inception_v3(weights="DEFAULT", transform_input=True).to(DEVICE)
    inception.eval()
    inception.fc = torch.nn.Identity()

    real_features, fake_features = inception_get_features(inception, real_images), inception_get_features(inception, fake_images)

    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False) + 1e-6 * np.eye(real_features.shape[1])
    sigma_fake = np.cov(fake_features, rowvar=False) + 1e-6 * np.eye(fake_features.shape[1])

    mean_term = np.sum((mu_real - mu_fake) ** 2) 
    covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = mean_term + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(fid_score)


def save_spectra_comparison(target_spectra, predicted_spectra, output_path, pearson_corr, spearman_corr):
    title = f"Target vs Predicted IR Spectra\nPearson r: {pearson_corr:.4f}, Spearman ρ: {spearman_corr:.4f}"
    plt.figure(figsize=(12, 6))
    x = np.arange(len(target_spectra))
    
    plt.plot(x, target_spectra, label="Target Spectra", alpha=0.7)
    plt.plot(x, predicted_spectra, label="Predicted Spectra", alpha=0.7)
    plt.xlabel('Wavenumber Index')
    plt.ylabel('Intensity')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
