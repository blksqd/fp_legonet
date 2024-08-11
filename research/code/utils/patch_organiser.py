import os
import shutil
import re

# Define the path to the directory containing the patches
source_dir = "/Users/andreshofmann/Desktop/Studies/Uol/7t/FP/stage_2/Images/img_patches/isic_mel_patches"

# Ensure the source directory exists
if not os.path.exists(source_dir):
    print(f"The directory {source_dir} does not exist.")
    exit(1)

# Loop to create directories 1 to 16
for i in range(1, 17):
    # Create the folder name
    folder_name = str(i)  
    
    # Create the full path for the new directory
    folder_path = os.path.join(source_dir, folder_name)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Loop through all files in the source directory
for file_name in os.listdir(source_dir):
    # Check if the file name matches the pattern
    match = re.match(r"^ISIC_\d+_patch_(\d+)\.jpg$", file_name)
    if match:
        try:
            # Extract the patch number from the file name
            patch_number = int(match.group(1))
            
            # Define the full path to the patch file
            patch_path = os.path.join(source_dir, file_name)
            
            # Define the target directory based on the patch number
            target_dir = os.path.join(source_dir, str(patch_number)) 
            
            # Ensure the target directory exists
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            # Define the new path for the patch file
            new_path = os.path.join(target_dir, file_name)
            
            # Move the file to the new directory
            shutil.move(patch_path, new_path)
        except ValueError:
            print(f"Skipping file {file_name}, does not match expected pattern.")

print("Files have been moved successfully.")