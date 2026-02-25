import matplotlib.pyplot as plt
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

folder_paths = ["C:/Users/ttjrb/Desktop/breasttumor/malignant", "C:/Users/ttjrb/Desktop/breasttumor/benign"] # Paths to the folders

size = 128 # we will conver our 483x560x3 images to 128x128x1

images = [] #to store the original images
masks = [] #to store the masks

found_mask = False # flag to handle multiple masks for the same image

for folder_path in folder_paths:

    # Loop through all files in the current folder (sorted for consistency)
    for file_path in sorted(glob(folder_path + "/*")):
        # Load and resize the image
        img = cv2.imread(file_path)
        img = cv2.resize(img, (size, size)) # Resize
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Convert RGB to grayscale (1 channel)
        img = img / 255.0 # Normalize to [0,1] scale

        if "mask" in file_path: # Checks if it is a "mask"
            if found_mask: #
                # Combine with the previous mask
                masks[-1] += img
                # Ensure binary output (0 or 1)
                masks[-1] = np.where(masks[-1] > 0.5, 1.0, 0.0)
            else:
                masks.append(img) # Adds the first mask to the list
                found_mask = True
        else:
            images.append(img) # Adds original image to the list
            found_mask = False

# Convert lists to NumPy arrays
X = np.array(images) # Create an array of all original images
y = np.array(masks) # Create an array of all masked imaged (Ground truth)

X = np.expand_dims(X, -1)
y = np.expand_dims(y, -1)

print(f"X shape: {X.shape} | y shape: {y.shape}")
# Split images into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

