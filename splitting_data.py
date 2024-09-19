import os
import shutil
import random

# Set the paths
dataset_dir = r'Labelled Dataset'  
train_dir = r'Labelled Dataset\train'  
test_dir = r'Labelled Dataset\test'  

# Create train and test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Iterate through each category (subfolder)
categories = ['animals', 'man_made', 'nature', 'people']  # List your categories
for category in categories:
    category_path = os.path.join(dataset_dir, category)
    
    # Create category directories in train and test
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)
    
    # Get all image files in the category
    images = os.listdir(category_path)
    random.shuffle(images)  # Shuffle the images
    
    # Calculate split index
    split_index = int(len(images) * 0.8)  # 80% for training
    
    # Split images into training and testing
    train_images = images[:split_index]
    test_images = images[split_index:]
    
    # Move images to the training and testing directories
    for img in train_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(train_dir, category, img))
    
    for img in test_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(test_dir, category, img))

print("Dataset split into training and testing sets.")