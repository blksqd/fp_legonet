import os
import shutil
import pandas as pd

# Load the metadata CSV file
metadata = pd.read_csv('/Users/andreshofmann/Desktop/Studies/Uol/7t/FP/stage_1/Data/ISIC/ISIC_2020_Training_GroundTruth.csv')

# Extract the IDs of images diagnosed as processed
processed_images = metadata[metadata['benign_malignant'] == 'malignant']['image_name'].tolist()

# Define source and target directories
source_dir = '/Users/andreshofmann/Desktop/Studies/Uol/7t/FP/stage_1/Data/ISIC/train'
target_dir = '/Users/andreshofmann/Desktop/Studies/Uol/7t/FP/stage_2/Images/isic_mel'

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Move processed images to the target directory
for image_id in processed_images:
    image_file = f"{image_id}.jpg"
    source_path = os.path.join(source_dir, image_file)
    target_path = os.path.join(target_dir, image_file)
    
    if os.path.exists(source_path):
        shutil.move(source_path, target_path)
        print(f"Moved: {image_file}")
    else:
        print(f"Image not found: {image_file}")
