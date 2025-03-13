import os
import joblib
import cv2
import numpy as np
from skimage.feature import hog
from tkinter import Tk, Label, Button, StringVar, OptionMenu, filedialog
import importlib.util  # To import models dynamically

# Define model paths using raw strings
model_paths = {
    "SVM (55%)": ("svm_model.pkl", "svm_model_scaler.pkl"),
    "CNN (85%)": (r"Comparative Analysis\svm_cnn_model.pkl", r"Comparative Analysis\svm_cnn_model_scaler.pkl"),
    "K-Nearest Neighbors (KNN) (41%)": (r"Comparative Analysis\knn_model.pkl", r"Comparative Analysis\knn_scaler.pkl"),
    "RBF Kernel SVM (67%)": (r"Comparative Analysis\svm_rbf_model.pkl", r"Comparative Analysis\rbf_scaler.pkl"),
    "Random Forest (63%)": (r"Comparative Analysis\random_forest_model.pkl", r"Comparative Analysis\random_forest_scaler.pkl"),
    "Decision Tree (42%)": (r"Comparative Analysis\decision_tree_model.pkl", r"Comparative Analysis\decision_tree_scaler.pkl"),
}

# Function to dynamically load a module from a file path
def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Function to extract HOG, SIFT, and Color Histogram features
def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

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

    # Combine all features
    return np.concatenate((hog_features, sift_features, color_hist))

# Function to classify an image using the selected model
def upload_and_classify():
    chosen_model = model_var.get()
    if chosen_model == "Select Model":
        result.set("Please select a valid model.")
        return

    # Select an image
    image_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not image_path:
        result.set("No image selected.")
        return

    # Get model and scaler file paths
    model_file, scaler_file = model_paths[chosen_model]

    try:
        # Load the model and scaler
        model = joblib.load(model_file)  # Load trained model
        scaler = joblib.load(scaler_file) if os.path.exists(scaler_file) else None  # Load scaler if available

        # Extract features
        if chosen_model == "CNN (85%)":
            cnn_script_path = r"Comparative Analysis\feature extraction using cnn.py"
            cnn_module = load_module_from_path("cnn_module", cnn_script_path)
            features = cnn_module.extract_features_with_cnn(image_path)  # ✅ Correct function call for CNN
        else:
            features = extract_features(image_path)

        if features is None:
            result.set("Error processing image.")
            return

        # Normalize features
        features_normalized = scaler.transform([features]) if scaler else features.reshape(1, -1)

        # 🛑 FIXED: Ensure `model` is used for prediction
        prediction = model.predict(features_normalized)  # ✅ Use the model, NOT the scaler!

        class_names = ['animals', 'man_made', 'nature', 'people']
        predicted_class = class_names[int(prediction[0])]

        # Display result
        result.set(f"Classification Result: {predicted_class}")

    except FileNotFoundError:
        result.set("Model or scaler file not found.")
    except Exception as e:
        result.set(f"Error: {str(e)}")

# GUI Setup
root = Tk()
root.title("Image Classification Model Selector")
root.geometry("400x250")

# Instruction Label
Label(root, text="Select a model and upload an image for classification.").pack(pady=10)

# Dropdown menu for model selection
model_var = StringVar(root)
model_var.set("Select Model")
model_menu = OptionMenu(root, model_var, *model_paths.keys())
model_menu.pack(pady=10)

# Result label
result = StringVar(root)
result_label = Label(root, textvariable=result, wraplength=300, justify="left")
result_label.pack(pady=10)

# Upload Button
Button(root, text="Upload Image", command=upload_and_classify).pack(pady=10)

# Start GUI loop
root.mainloop()
