import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib  # to save and load the model
import matplotlib.pyplot as plt

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
    sift_features = np.mean(descriptors, axis=0) if descriptors is not None else np.zeros((128,))

    # Color Histogram
    color_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    color_hist = cv2.normalize(color_hist, color_hist).flatten()

    # LBP Feature Extraction
    lbp = local_binary_pattern(gray_image, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    hist = hist.astype("float")
    hist /= hist.sum()  # Normalize histogram

    # Combine all features
    features = np.concatenate((hog_features, sift_features, color_hist, hist))
    print(f"Features extracted for {image_path}")
    return features

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

# Function to train the SVM model
def train_model(X_train, y_train):
    print("Training the SVM model with RBF kernel...")
    model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')  # Experiment with C and gamma values
    model.fit(X_train, y_train)
    return model

# Function to classify a new image
def classify_new_image(image_path, model, scaler):
    features = extract_features(image_path)
    if features is None:
        return "Error in extracting features."

    # Scale the features using the loaded scaler
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    
    # Map the prediction back to the category
    categories = ['animals', 'man_made', 'nature', 'people']
    return categories[prediction[0]]

# Main script
if __name__ == "__main__":
    print("Starting image classification using SVM with RBF kernel...")

    # Check if model and scaler exist
    model_file = 'svm_rbf_model.pkl'
    scaler_file = 'scaler.pkl'
    
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        print("Loading the saved model and scaler...")
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
    else:
        print("Model and scaler not found. Training the model...")
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
        split_index = int(len(y) * 0.8)
        X_train = scaler.fit_transform(X[:split_index])  # 80% for training
        y_train = y[:split_index]
        X_test = scaler.transform(X[split_index:])        # 20% for testing
        y_test = y[split_index:]

        print(f"Training data size: {X_train.shape}")
        print(f"Test data size: {X_test.shape}")
        
        # Train the SVM model
        model = train_model(X_train, y_train)

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
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(['animals', 'man_made', 'nature', 'people']))
        plt.xticks(tick_marks, ['animals', 'man_made', 'nature', 'people'], rotation=45)
        plt.yticks(tick_marks, ['animals', 'man_made', 'nature', 'people'])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    # Example of classifying a new image
    new_image_path = 'Giraffe.png'  # Replace with the path to your new image
    predicted_class = classify_new_image(new_image_path, model, scaler)
    print(f"The new image is classified as: {predicted_class}")
