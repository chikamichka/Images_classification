import os
import pandas as pd
import requests
from urllib.parse import urlsplit

# Function to download and save image
def download_image(url, folder):
    try:
        # Get the image content
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get the image filename from the URL
        filename = os.path.basename(urlsplit(url).path)
        filepath = os.path.join(folder, filename)

        # Save the image to the folder
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

# Function to load images from CSV
def load_images_from_csv(csv_file, folder):
    # Read CSV file
    df = pd.read_csv(csv_file)

    # Ensure the destination folder exists
    os.makedirs(folder, exist_ok=True)

    # Loop through each image URL in the CSV and download it
    for index, row in df.iterrows():
        image_url = row['Image URL']  # Assuming the CSV column with URLs is named 'image_url'
        download_image(image_url, folder)

# Example usage
csv_file = 'tweets.csv'  # Path to your CSV file
destination_folder = 'downloaded_images'  # Folder where images will be saved

load_images_from_csv(csv_file, destination_folder)
