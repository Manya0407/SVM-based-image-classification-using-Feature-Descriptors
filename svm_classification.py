import os
import numpy as np
import cv2
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from skimage.feature import hog
from concurrent.futures import ThreadPoolExecutor

# Load the EfficientNet model globally to avoid reloading for every image
efficientnet_model = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')

# Function to extract features from an image
def extract_features(image_path):
    print(f"Extracting features from {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return None

    # Resize image to a fixed size
    image_resized = cv2.resize(image, (128, 128))

    # HOG Feature Extraction (CPU-bound)
    hog_features = hog(image_resized, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False, channel_axis=-1)

    # EfficientNet Feature Extraction (GPU-bound)
    image_preprocessed = preprocess_input(image_resized.astype(np.float32))  # Preprocess for EfficientNet
    image_expanded = np.expand_dims(image_preprocessed, axis=0)  # Expand dimensions for model input
    efficientnet_features = efficientnet_model.predict(image_expanded)

    # Combine HOG and EfficientNet features
    features = np.concatenate((hog_features, efficientnet_features.flatten()))
    print(f"Features extracted for {image_path}")
    return features

# Function to process images in parallel
def process_image(image_data, data_dir, label_map):
    category, img_file, is_train = image_data
    img_path = os.path.join(data_dir, 'train' if is_train else 'test', category, img_file)
    features = extract_features(img_path)
    if features is not None:
        return features, label_map[category]
    return None

# Load dataset from train and test directories
def load_data(data_dir):
    print("Loading data...")
    X = []
    y = []
    categories = ['animals', 'man_made', 'nature', 'people']
    label_map = {category: idx for idx, category in enumerate(categories)}

    image_data = []

    # Load training data
    for category in categories:
        category_path_train = os.path.join(data_dir, 'train', category)
        print(f"Loading training data from category: {category}")
        for img_file in os.listdir(category_path_train):
            image_data.append((category, img_file, True))

    # Load testing data
    for category in categories:
        category_path_test = os.path.join(data_dir, 'test', category)
        print(f"Loading testing data from category: {category}")
        for img_file in os.listdir(category_path_test):
            image_data.append((category, img_file, False))

    # Parallel processing of images
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda img: process_image(img, data_dir, label_map), image_data))

    # Filter out None results and separate features and labels
    results = [result for result in results if result is not None]
    X, y = zip(*results)

    print("Data loading complete.")
    return np.array(X), np.array(y)

# Main script
if __name__ == "__main__":
    print("Starting image classification using SVM...")

    data_dir = 'Labelled Dataset'  # Path to your dataset directory

    # Load the data
    X, y = load_data(data_dir)

    # Print data information
    print(f"Total images processed: {len(y)}")
    print(f"Number of features per image: {X.shape[1]}")

    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()

    # Training on 80%, testing on 20%
    split_index = len(y) // 5 * 4
    X_train = scaler.fit_transform(X[:split_index])  # 80% for training
    y_train = y[:split_index]
    X_test = scaler.transform(X[split_index:])       # 20% for testing
    y_test = y[split_index:]

    print(f"Training data size: {X_train.shape}")
    print(f"Test data size: {X_test.shape}")

    # Train the SVM model
    print("Training the SVM model...")
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)

    print("Model training complete.")

    # Predictions
    print("Making predictions on test data...")
    y_pred = model.predict(X_test)

    # Evaluation
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=['animals', 'man_made', 'nature', 'people']))

    print("Classification process complete.")
