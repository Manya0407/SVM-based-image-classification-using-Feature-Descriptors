import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib  # to save and load the model
import time

# Function to extract features from an image
def extract_features(image_path):
    print(f"Extracting features from {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return None

    # Resize image to a fixed size
    image = cv2.resize(image, (128, 128))

    # HOG Feature Extraction
    hog_features = hog(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False, channel_axis=-1)

    # SIFT Feature Extraction
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    if descriptors is None:
        sift_features = np.zeros((128,))  # Default if no keypoints detected
    else:
        sift_features = np.mean(descriptors, axis=0)  # Average of SIFT descriptors

    # Color Histogram
    color_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_hist = cv2.normalize(color_hist, color_hist).flatten()

    # Combine all features
    features = np.concatenate((hog_features, sift_features, color_hist))
    print(f"Features extracted for {image_path}")
    return features

# Function to classify a new image using the trained model
def classify_new_image(image_path, model, scaler):
    print(f"Classifying new image: {image_path}")
    
    # Extract features
    features = extract_features(image_path)
    if features is None:
        return "Error in image processing"
    
    # Standardize the features using the training data scaler
    features_normalized = scaler.transform([features])  # Reshape and standardize
    
    # Predict class using trained Random Forest model
    prediction = model.predict(features_normalized)
    class_names = ['animals', 'man_made', 'nature', 'people']
    predicted_class = class_names[int(prediction[0])]
    
    print(f"Predicted class: {predicted_class}")
    return predicted_class

# Load dataset from train and test directories
def load_data(data_dir):
    print("Loading data...")
    X = []
    y = []
    categories = ['animals', 'man_made', 'nature', 'people']
    label_map = {category: idx for idx, category in enumerate(categories)}

    # Load training data
    for category in categories:
        category_path_train = os.path.join(data_dir, 'train', category)
        print(f"Loading training data from category: {category}")

        for img_file in os.listdir(category_path_train):
            img_path = os.path.join(category_path_train, img_file)
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(label_map[category])

    # Load testing data
    for category in categories:
        category_path_test = os.path.join(data_dir, 'test', category)
        print(f"Loading testing data from category: {category}")

        for img_file in os.listdir(category_path_test):
            img_path = os.path.join(category_path_test, img_file)
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(label_map[category])

    print("Data loading complete.")
    return np.array(X), np.array(y)

# Main script
if __name__ == "__main__":
    print("Starting image classification using Random Forest...")

    start_time = time.time()  # Start time

    # Check if model and scaler exist
    model_file = r'Comparative Analysis\random_forest_model.pkl'
    scaler_file = r'Comparative Analysis\random_forest_scaler.pkl'
    
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        print("Loading the saved model and scaler...")
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
    else:
        print("Model and scaler not found. Training the model...")
        data_dir = r'.\Labelled Dataset'  # Path to your dataset directory
        
        # Load the data
        X, y = load_data(data_dir)
        
        # Print data information
        print(f"Total images processed: {len(y)}")
        print(f"Number of features per image: {X.shape[1]}")

        # Standardize features
        print("Standardizing features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data into training and test sets (80-20)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        print(f"Training data size: {X_train.shape}")
        print(f"Test data size: {X_test.shape}")
        
        # Train the Random Forest model
        print("Training the Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save the model and scaler
        joblib.dump(model, model_file)
        joblib.dump(scaler, scaler_file)

        print(f"Model training complete. Model saved as '{model_file}'.")

        # Predictions
        print("Making predictions on test data...")
        y_pred = model.predict(X_test)

        # Evaluation
        print("Classification report:")
        print(classification_report(y_test, y_pred, target_names=['animals', 'man_made', 'nature', 'people']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)

        # Calculate and print the total runtime
        end_time = time.time()
        total_time_seconds = end_time - start_time
        minutes = int(total_time_seconds // 60)
        seconds = total_time_seconds % 60
        print(f"Total time taken to run the script: {minutes} minutes and {seconds:.2f} seconds")

    # Example of classifying a new image
    new_image_path = r'.\Unseen Image\building.jpeg'  # Replace with the path to your new image
    predicted_class = classify_new_image(new_image_path, model, scaler)
    print(f"The new image is classified as: {predicted_class}")