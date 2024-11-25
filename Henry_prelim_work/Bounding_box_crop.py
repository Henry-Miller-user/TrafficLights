import json
import os
import cv2

# Define paths
image_dir = "/Users/henry/Downloads/Deep_learning_p/train_dataset/train_images"
json_path = "/Users/henry/Downloads/Deep_learning_p/train_dataset/train.json"
output_dir = "/Users/henry/Downloads/Deep_learning_p/Cropped_Traffic_Lights"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Process annotations
for annotation in data['annotations']:
    # Normalize filename path
    filename = annotation['filename'].replace("\\", "/")
    image_path = os.path.join(image_dir, os.path.basename(filename))

    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        continue

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        continue

    # Process each inbox bounding box
    for idx, light in enumerate(annotation['inbox']):
        color = light.get('color', 'unknown')  # Default to 'unknown' if color not present
        bbox = light.get('bndbox', None)

        if not bbox:
            print(f"Missing bounding box for: {filename}, light index {idx}")
            continue

        # Extract bounding box coordinates
        xmin, ymin, xmax, ymax = map(int, [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']])

        # Crop the image
        cropped_image = image[ymin:ymax, xmin:xmax]

        # Save cropped image
        output_filename = f"{os.path.splitext(os.path.basename(filename))[0]}_{idx}_{color}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, cropped_image)

        print(f"Saved cropped image: {output_path}")

print("Processing complete.")
