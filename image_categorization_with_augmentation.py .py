import os
import json
import shutil
import cv2
import numpy as np

# Load the JSON file
json_file_path = 'all.json'  # Update with your JSON file path
jpg_images_folder = 'raw'    # Folder containing the raw images
output_folder = 'Labelled Dataset'  # Folder to categorize images into

# Function to perform image augmentation for animal images
def augment_animal_image(image, image_name, subject_folder):
    # Augmentations
    augmentations = [
        ('_flipped', cv2.flip(image, 1)),                         # Horizontal flip
        ('_rotated_90', cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),  # 90 degree rotation
        ('_rotated_180', cv2.rotate(image, cv2.ROTATE_180)),         # 180 degree rotation
        ('_bright', cv2.convertScaleAbs(image, alpha=1.2, beta=30))   # Brightness adjustment
    ]
    
    # Apply and save each augmentation
    for suffix, aug_image in augmentations:
        aug_image_name = f"{image_name}{suffix}.jpg"
        aug_image_path = os.path.join(subject_folder, aug_image_name)
        cv2.imwrite(aug_image_path, aug_image)
        print(f"Saved augmented image: {aug_image_name}")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load JSON data
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Create a mapping from image names to their subjects (all classes in the JSON)
name_to_subject = {item['name']: item['categories']['subject'] for item in data}

# Loop through each JPG file in the directory
for image_name, subject in name_to_subject.items():
    # Create subject folder (category) if it doesn't exist
    subject_folder = os.path.join(output_folder, subject)
    os.makedirs(subject_folder, exist_ok=True)

    # Construct the JPG file name
    jpg_file_name = f"{image_name}.jpg"

    # Source and destination paths
    source_path = os.path.join(jpg_images_folder, jpg_file_name)
    destination_path = os.path.join(subject_folder, jpg_file_name)

    # Move the file if it exists
    if os.path.exists(source_path):
        # Move the image to its respective category folder
        shutil.move(source_path, destination_path)
        print(f"Moved {jpg_file_name} to {subject_folder}")

        # Check if the category is 'animals' and perform augmentation
        if subject.lower() == 'animals':
            image = cv2.imread(destination_path)
            augment_animal_image(image, image_name, subject_folder)
    else:
        print(f"File {jpg_file_name} not found.")

print("Categorization and augmentation (for animals) completed.")
