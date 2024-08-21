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
target_dir = config['paths']['model_save_path']
patch_save_dir = config['paths']['patch_save_dir']  # Directory where patches will be saved
patch_csv_file = config['paths']['patch_csv_file']  # CSV file to save patch metadata

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
def preprocess_image(image_path):
    # Use the crop_and_center_image function from the refactored image_cropper.py
    cropped_image = crop_and_center_image(image_path)
    if cropped_image is None:
        raise ValueError(f"Could not process image: {image_path}")
    return cropped_image

# Step 2: Patch Division and Labeling
def divide_image_into_patches(image):
    # Divide the 256x256 image into 16 non-overlapping patches of 64x64 pixels
    patches = generate_patches(image, patch_size=(64, 64))
    return patches

# Step 3: Patch Generation and Augmentation using cGAN
def augment_patches_with_cgan(patches, num_augmentations=16):
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
def aggregate_classifications(classifications):
    majority_vote = np.mean(classifications > 0.5)
    return 1 if majority_vote > 0.5 else 0

# Main Pipeline Execution
def run_pipeline(image_path):
    # Step 1: Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    
    # Step 2: Divide the image into patches
    patches = divide_image_into_patches(preprocessed_image)
    
    # Step 3: Augment the patches using cGAN
    augmented_patches = augment_patches_with_cgan(patches)
    
    # Step 4: Extract features from patches
    features = extract_features(augmented_patches)
    
    # Step 5: Classify patches
    classifications = classify_patches(features)
    
    # Step 6: Aggregate patch classifications
    final_classification = aggregate_classifications(classifications)
    
    # Output the final classification
    if final_classification == 1:
        print("The image is classified as malignant.")
    else:
        print("The image is classified as benign.")
    
    return final_classification

# Preprocessing: Move the necessary images based on metadata
move_processed_images(metadata_csv_path, source_dir, target_dir, label='malignant')

# Generate patches from directory and save the metadata
generate_patches_from_directory(source_dir, patch_save_dir, patch_csv_file)

# Organize the patches into directories based on their patch number
organise_patches_into_directories(patch_save_dir)

# Example usage of run_pipeline (Main Processing Pipeline)
final_result = run_pipeline(input_image_path)