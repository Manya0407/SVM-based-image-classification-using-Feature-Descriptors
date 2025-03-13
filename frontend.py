import os
import subprocess
from tkinter import Tk, Label, Button, StringVar, OptionMenu, filedialog
import importlib  # To dynamically import model files

# Mapping each model to its respective Python module and function
model_files = {
    "SVM (55%)": "svm_classification",
    "CNN (85%)": "Comparative Analysis.feature_extraction_using_cnn",
    "K-Nearest Neighbors (KNN) (41%)": "Comparative Analysis.KNN",
    "RBF Kernel SVM (67%)": "Comparative Analysis.RBF",
    "Random Forest (63%)": "Comparative Analysis.random_forest",
    "Decision Tree (42%)": "Comparative Analysis.decision_trees"
}

# Initialize GUI window
root = Tk()
root.title("Image Classification Model Selector")
root.geometry("400x200")

# Label for instructions
Label(root, text="Select a model and upload an image for classification.").pack(pady=10)

# Dropdown menu to select the model
model_var = StringVar(root)
model_var.set("Select Model")  # Default value

model_menu = OptionMenu(root, model_var, *model_files.keys())
model_menu.pack(pady=10)

# Result label
result = StringVar(root)
result_label = Label(root, textvariable=result, wraplength=300, justify="left")
result_label.pack(pady=10)

# Function to open file dialog to select an image and run the chosen model
def upload_and_classify():
    # Get the chosen model
    chosen_model = model_var.get()

    if chosen_model == "Select Model":
        result.set("Please select a valid model.")
        return

    # Open file dialog for image upload
    image_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if not image_path:
        result.set("No image selected.")
        return

    # Dynamically import the corresponding model module
    try:
        model_module = importlib.import_module(model_files[chosen_model])
        # Assume each model has a classify function accepting image_path
        classification_result = model_module.classify_new_image(image_path)  # Replace with the correct function name
        result.set(f"Classification Result:\n{classification_result}")
    except ImportError:
        result.set("Error importing the model.")
    except AttributeError:
        result.set("Model does not have the expected classification function.")
    except Exception as e:
        result.set(f"Error: {str(e)}")

# Button to upload image and classify
Button(root, text="Upload Image", command=upload_and_classify).pack(pady=10)

# Run the GUI loop
root.mainloop()
