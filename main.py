import os
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Path to the images folder (update this path if necessary)
image_directory = 'downloaded_images'

# Load the pre-trained Vision Transformer (ViT) model and processor
model_name = "google/vit-base-patch16-224-in21k"  # Pre-trained ViT model
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

# Initialize a list to store results
results = []

# Function to load and preprocess the image
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")  # Ensure image is in RGB mode
    return processor(images=img, return_tensors="pt")

# Function to classify an image
def classify_image(img_path):
    inputs = preprocess_image(img_path)
    with torch.no_grad():
        logits = model(**inputs).logits  # Get model's raw predictions
    predicted_class_idx = logits.argmax(-1).item()  # Get the index of the predicted class
    return predicted_class_idx

# Iterate over all images in the directory and classify them
for img_name in os.listdir(image_directory):
    img_path = os.path.join(image_directory, img_name)
    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure it's an image file
        predicted_class_idx = classify_image(img_path)

        # Assuming the model has two classes (healthy, unhealthy)
        # You may need to adjust this based on your specific model or your labels
        if predicted_class_idx == 0:
            category = 'healthy'
        else:
            category = 'unhealthy'

        # Store the results
        results.append({
            'image': img_name,
            'category': category,
            'class_id': predicted_class_idx
        })

        # Display the image with the predicted label
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(f"Image: {img_name} - Predicted: {category}")
        plt.axis('off')  # Hide the axes
        plt.show()

# Convert results to DataFrame and save as CSV
df = pd.DataFrame(results)
df.to_csv('classification_results.csv', index=False)

print("Classification complete. Results saved in 'classification_results.csv'.")
