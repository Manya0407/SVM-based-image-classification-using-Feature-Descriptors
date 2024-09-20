import os
import json
import shutil

# Load the JSON file
json_file_path = 'all.json'  # Update with your JSON file path
jpg_images_folder = 'raw'  
output_folder = 'Labelled Dataset'  

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load JSON data
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Create a mapping from image names to subjects
name_to_subject = {item['name']: item['categories']['subject'] for item in data}

# Loop through each JPG file in the directory
for image_name, subject in name_to_subject.items():
    # Create subject folder if it doesn't exist
    subject_folder = os.path.join(output_folder, subject)
    os.makedirs(subject_folder, exist_ok=True)

    # Construct the JPG file name
    jpg_file_name = f"{image_name}.jpg"

    # Source and destination paths
    source_path = os.path.join(jpg_images_folder, jpg_file_name)
    destination_path = os.path.join(subject_folder, jpg_file_name)

    # Move the file if it exists
    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)
        print(f"Moved {jpg_file_name} to {subject_folder}")
    else:
        print(f"File {jpg_file_name} not found.")

print("Categorization completed.")
