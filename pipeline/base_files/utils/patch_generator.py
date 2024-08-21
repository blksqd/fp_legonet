import os
import cv2
import pandas as pd

# Define source and target directories
source_dir = '/Users/andreshofmann/Desktop/Studies/Uol/7t/FP/stage_2/Images/cropped_isic_mel_imgs'
target_dir = '/Users/andreshofmann/Desktop/Studies/Uol/7t/FP/stage_2/Images/img_patches/isic_mel_patches'
csv_file_path = '/Users/andreshofmann/Desktop/Studies/Uol/7t/FP/stage_2/csv_files/isic_mel_patches_info.csv'

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Initialize a list to hold patch information for the CSV (important for the tracking of images!)
patch_info_list = []

def save_patches(image_path, target_size=(256, 256), patch_size=(64, 64)):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Ensure the image is the target size
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # Get the dimensions of the patches
    patch_h, patch_w = patch_size

    # Initialize a patch counter
    patch_counter = 1

    # Loop to create patches
    for y in range(0, target_size[1], patch_h):
        for x in range(0, target_size[0], patch_w):
            # Extract the patch
            patch = image[y:y + patch_h, x:x + patch_w]

            # Define the patch file name
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            patch_file_name = f"{base_name}_patch_{patch_counter}.jpg"

            # Define the patch save path
            patch_save_path = os.path.join(target_dir, patch_file_name)

            # Save the patch
            cv2.imwrite(patch_save_path, patch)

            # Append patch information to the list <---- Lists will need to be mixed later
            patch_info_list.append({
                'image_id': base_name,
                'patch_id': patch_file_name
            })

            # Increment the patch counter
            patch_counter += 1

# Process each image
for image_file in os.listdir(source_dir):
    if image_file.endswith('.jpg'):
        image_path = os.path.join(source_dir, image_file)
        save_patches(image_path)
        print(f"Subdivided and saved patches for: {image_file}")

# Create a DataFrame from the patch information list
patch_info_df = pd.DataFrame(patch_info_list)

# Save the DataFrame to a CSV file
patch_info_df.to_csv(csv_file_path, index=False)
print(f"Saved patch information to: {csv_file_path}")