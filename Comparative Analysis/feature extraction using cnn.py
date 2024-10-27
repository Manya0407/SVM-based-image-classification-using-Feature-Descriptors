import os
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load pre-trained ResNet50 model for feature extraction
cnn_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Categories for classification
categories = ['animals', 'man_made', 'nature', 'people']

# Function to extract CNN features for a given image
def extract_features_with_cnn(image_path):
    print(f"Extracting CNN features from {image_path}")
    img = image.load_img(image_path, target_size=(128, 128))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    
    cnn_features = cnn_model.predict(img_data)
    print(f"Features extracted for {image_path}")
    return cnn_features.flatten()

# Function to load dataset and process images with parallel processing
def load_data(data_dir):
    print("Loading data...")
    X = []
    y = []
    label_map = {category: idx for idx, category in enumerate(categories)}

    # Helper function to process images in parallel
    def process_image(img_path, label):
        features = extract_features_with_cnn(img_path)
        return features, label

    # List to store parallel processing tasks
    futures = []
    with ThreadPoolExecutor() as executor:
        # Load training data
        for category in categories:
            category_path_train = os.path.join(data_dir, 'train', category)
            label = label_map[category]
            print(f"Loading training data from category: {category}")
            for img_file in os.listdir(category_path_train):
                img_path = os.path.join(category_path_train, img_file)
                futures.append(executor.submit(process_image, img_path, label))
        
        # Load testing data
        for category in categories:
            category_path_test = os.path.join(data_dir, 'test', category)
            label = label_map[category]
            print(f"Loading testing data from category: {category}")
            for img_file in os.listdir(category_path_test):
                img_path = os.path.join(category_path_test, img_file)
                futures.append(executor.submit(process_image, img_path, label))
        
        # Collect results as they complete
        for future in as_completed(futures):
            features, label = future.result()
            if features is not None:
                X.append(features)
                y.append(label)

    print("Data loading complete.")
    return np.array(X), np.array(y)

# Main script for training and evaluating the SVM model
if __name__ == "__main__":
    print("Starting image classification using CNN features and SVM...")

    # Set the dataset directory
    data_dir = r"C:\Users\sajal\Desktop\ML SVM PROJECT\SVM-based-image-classification-using-Feature-Descriptors\Labelled Dataset"  # Path to your dataset directory

    # Load the data with parallel processing
    X, y = load_data(data_dir)
    
    # Print data information
    print(f"Total images processed: {len(y)}")
    print(f"Number of features per image: {X.shape[1]}")

    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    
    # Split data for training (80%) and testing (20%)
    split_index = len(y) // 5 * 4
    X_train = scaler.fit_transform(X[:split_index])  # 80% for training
    y_train = y[:split_index]
    X_test = scaler.transform(X[split_index:])        # 20% for testing
    y_test = y[split_index:]

    print(f"Training data size: {X_train.shape}")
    print(f"Test data size: {X_test.shape}")
    
    # Train the SVM model
    print("Training the SVM model...")
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Predictions on test data
    print("Making predictions on test data...")
    y_pred = model.predict(X_test)

    # Evaluation
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=categories))