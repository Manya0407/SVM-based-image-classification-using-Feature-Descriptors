import os
import cv2

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

# Directory and image details
output_folder = 'Labelled Dataset'  # Folder containing categorized images

# Loop through each subject folder in the output directory
for subject in os.listdir(output_folder):
    subject_folder = os.path.join(output_folder, subject)

    # Process only animal subjects
    if subject.lower() == 'animals':
        for image_name in os.listdir(subject_folder):
            if image_name.endswith('.jpg'):
                image_path = os.path.join(subject_folder, image_name)
                image = cv2.imread(image_path)
                
                # Perform augmentation
                augment_animal_image(image, image_name[:-4], subject_folder)  # Exclude '.jpg' extension

print("Augmentation for animal images completed.")