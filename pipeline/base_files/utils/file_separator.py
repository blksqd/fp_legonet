import os
import shutil
import pandas as pd

def move_processed_images(metadata_csv_path, source_dir, target_dir, label='malignant'):
    # Load the metadata CSV file
    metadata = pd.read_csv(metadata_csv_path)

    # Extract the IDs of images diagnosed as processed based on the specified label
    processed_images = metadata[metadata['benign_malignant'] == label]['image_name'].tolist()

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