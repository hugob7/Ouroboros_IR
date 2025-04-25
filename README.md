# Ouroboros IR: Generative-Predictive Model with H&E Images and IR Spectra

Ouroboros IR is a generative-predictive model for generating synthetic H&E images from IR spectra and predicting IR spectra from real/synthetic H&E images. This code adapts the Ouroboros framework developed by Srijay Deshpande (https://github.com/Srijay/Ouroboros), originally designed for protein expression data, to work with IR spectra data.

The dataset consists of 4 GBM samples (A, B, C, D), divided into 9900 patches. The model is trained on a small subset of patches from a sample (30%) and tested on the remainder (70%). Repeated four times, once per sample.

## How to Run


### 1. Setup Environment
```bash
git clone https://github.com/hugob7/Ouroboros_IR.git
cd Ouroboros_IR

conda env create -f environment.yml

conda activate ouroboros_ir

pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. Preprocessing

The tissue_sample_analysis directory contains code for preprocessing the data before use in the Ouroboros framework. This consists of:

1. Processing IR spectra data
2. Registration
3. Extracting patches

To preprocess all four samples (A-D), run this bash script:

```bash
./tissue_sample_analysis/process_all_samples.sh
```

To preprocess a specific sample instead, run this bash script and specify the sample:

```bash
# Preprocess sample C
./tissue_sample_analysis/process_sample.sh C
```

Each of the Python preprocessing steps can be run individually for a sample if needbe.

```bash
# Process spectra data for sample C
python tissue_sample_analysis/ir_processing.py C

# Registration for sample C
python tissue_sample_analysis/registration.py C

# Extract patches from sample C
python tissue_sample_analysis/patch_extraction.py C
```

### 3. Training / Test Set Split

To split a sample's patches into train set and test set, run the below Python script:

```bash
python train_test_split_patches.py
```

This will go through each of the four samples individually. The sample's patches will be randomly split into train set (30%) and test set (70%), then all spectra are also standardised.

### 4. Training model

To train the model, run the below Python script:

```bash
python main.py --mode train
```

### 5. Testing and evaluating model

To test and evalute the model, run the below Python script:

```bash
python main.py --mode test --batch_size 1
```
