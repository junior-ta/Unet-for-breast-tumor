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

