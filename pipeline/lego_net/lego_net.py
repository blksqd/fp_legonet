"""
Make sure all models are in the right directory and called from the configuration file.
the models can be switched to no versions but this pipeline does not include a fine-tuning
block. This is to keep the pipeline as clean as possible. At this stage the model will
remain static and will not be updated. Current Val Acc. is 91%
"""

import os
import sys
import cv2
import numpy as np
import albumentations as A
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import json

# Add the path to the utilities directory to sys.path
sys.path.append('/Users/andreshofmann/Desktop/Studies/Uol/7t/FP/fp_legonet/pipeline/base_files/utils')

# Import utility functions
from image_cropper import crop_and_center_image
from file_separator import move_processed_images
from patch_generator import generate_patches_from_directory
from patch_organiser import organise_patches_into_directories

# Load the configuration file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Extract paths and parameters from config
input_image_path = config['paths']['input_image']
pretrained_cgan_model = config['paths']['pretrained_cgan_model']
pretrained_cvt_model = config['paths']['pretrained_cvt_model']
pretrained_cnn_model = config['paths']['pretrained_cnn_model']
metadata_csv_path = config['paths']['csv_path']
source_dir = config['paths']['img_dir']

# Set `patch_save_dir` and `patch_csv_file` based on expected usage
patch_save_dir = os.path.join(source_dir, "patches")  
patch_csv_file = os.path.join(source_dir, "patches_info.csv")  

# Load the pre-trained models
cgan_model = load_model(pretrained_cgan_model)
cvt_model = load_model(pretrained_cvt_model)
cnn_model = load_model(pretrained_cnn_model)

# Augmentation pipeline using albumentations with settings from config
augmentation_pipeline = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.Transpose(),
    A.OneOf([
        A.IAAAdditiveGaussianNoise() if config['data_augmentation']['use_gaussian_noise'] else A.NoOp(),
        A.GaussNoise() if config['data_augmentation']['use_gaussian_noise'] else A.NoOp(),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=0.2) if config['data_augmentation']['use_motion_blur'] else A.NoOp(),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=config['data_augmentation']['width_shift_range'], 
                       scale_limit=config['data_augmentation']['zoom_range'], 
                       rotate_limit=config['data_augmentation']['rotation_range'], 
                       p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.1) if config['data_augmentation']['use_grid_distortion'] else A.NoOp(),
        A.IAAPiecewiseAffine(p=0.3),
    ], p=0.2),
    A.OneOf([
        A.CLAHE(clip_limit=config['data_augmentation']['clahe_clip_limit']),
        A.IAASharpen() if config['data_augmentation']['use_sharpen'] else A.NoOp(),
        A.IAAEmboss() if config['data_augmentation']['use_emboss'] else A.NoOp(),
        A.RandomBrightnessContrast(brightness_limit=config['data_augmentation']['brightness_range'], 
                                   p=0.3) if config['data_augmentation']['use_random_brightness_contrast'] else A.NoOp(),
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
])

# Step 1: Preprocess the Image (Image Acquisition and Normalization)
"""
Lesion will be detected and zoom/center to use 80% of the total image space.
Later on, a control check can be added to make sure that is the injury is not
found, the image cannot be processed.
"""
def preprocess_image(image_path):
    cropped_image = crop_and_center_image(image_path)
    if cropped_image is None:
        raise ValueError(f"Could not process image: {image_path}")
    return cropped_image

# Step 2: Patch Division and Labeling using generate_patches_from_directory
"""Positions will be recorded in the file name to encode the position in the 
image group. This is an added feature to the embeddings created by the CvT"""
def divide_image_into_patches(source_dir, patch_save_dir, patch_csv_file):
    generate_patches_from_directory(source_dir, patch_save_dir, patch_csv_file)
    patches = []
    
    # Load generated patches
    for patch_file in os.listdir(patch_save_dir):
        patch = cv2.imread(os.path.join(patch_save_dir, patch_file))
        patches.append(patch)
    
    return np.array(patches)

# Step 3: Patch Generation and Augmentation using cGAN
"""
The number of augmentations was found to be best in the range from 8 - 12 
but this was established with a limited amount of hyperparameter search runs
"""
def augment_patches_with_cgan(patches, num_augmentations=12):
    augmented_patches = []
    for patch in patches:
        patch = np.expand_dims(patch, axis=0)
        for _ in range(num_augmentations):
            noise = np.random.normal(0, 1, (1, cgan_model.input_shape[1]))
            generated_patch = cgan_model.predict([noise, patch])
            augmented_patch = augmentation_pipeline(image=generated_patch[0])['image']
            augmented_patches.append(augmented_patch)
        augmented_patches.append(augmentation_pipeline(image=patch[0])['image'])
    return np.array(augmented_patches)

# Step 4: Feature Extraction using CvT
def extract_features(patches):
    features = cvt_model.predict(patches)
    return features

# Step 5: Patch-Level Classification using CNN
def classify_patches(features):
    classifications = cnn_model.predict(features)
    return classifications

# Step 6: Patch Classification Aggregation
"""
Change the majority vote % depending on how strict the system has to be.
For now is has been set to 80%
"""
def aggregate_classifications(classifications):
    majority_vote = np.mean(classifications > 0.5)
    return 1 if majority_vote > 0.8 else 0

# Main Pipeline Execution
def run_pipeline(input_image_path):
    # Preprocess the input image
    preprocessed_image = preprocess_image(input_image_path)
    
    # Save the preprocessed image to a temporary directory for patching
    temp_preprocessed_dir = os.path.join(source_dir, "temp_preprocessed")
    os.makedirs(temp_preprocessed_dir, exist_ok=True)
    temp_image_path = os.path.join(temp_preprocessed_dir, "preprocessed_image.jpg")
    cv2.imwrite(temp_image_path, preprocessed_image)
    
    # Divide the image into patches
    patches = divide_image_into_patches(temp_preprocessed_dir, patch_save_dir, patch_csv_file)
    
    # Augment the patches using the cGAN model and albumentations
    augmented_patches = augment_patches_with_cgan(patches)
    
    # Extract features from the augmented patches using the CvT model
    features = extract_features(augmented_patches)
    
    # Classify the patches using the CNN model
    classifications = classify_patches(features)
    
    # Aggregate the classifications to produce a final decision
    final_classification = aggregate_classifications(classifications)
    
    # Output the final classification result
    if final_classification == 1:
        print("The image is classified as malignant.")
    else:
        print("The image is classified as benign.")
    
    return final_classification

# Preprocessing: Move the necessary images based on metadata
move_processed_images(metadata_csv_path, source_dir, patch_save_dir, label='malignant')

# Generate patches from directory and save the metadata
organise_patches_into_directories(patch_save_dir)

# Instantiation of Main Processing Pipeline
final_result = run_pipeline(input_image_path)