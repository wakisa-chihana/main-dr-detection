import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load the trained model
model = load_model('effB3_CNN_DR_classifier.h5')

# Function to crop the image from gray areas
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # Image is too dark, return the original image
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

# Function to apply a circular crop and Gaussian blur
def circle_crop(img, sigmaX=10):
    """
    Create a circular crop around the image center and apply Gaussian Blur.
    """
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, _ = img.shape
    x = width // 2
    y = height // 2
    r = np.amin((x, y))
    
    # Create a mask for the circular crop
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), r, 1, thickness=-1)
    
    # Apply circular mask and blur
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    
    return img

# Define the image prediction function
def predict_image(img_path):
    try:
        # Check if the image file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file {img_path} not found")

        # Load the image in OpenCV format
        img = cv2.imread(img_path)
        
        # Apply circular crop and Gaussian blur processing
        processed_img = circle_crop(img, sigmaX=30)
        
        # Resize the processed image for the model input
        processed_img = cv2.resize(processed_img, (224, 224))

        # Convert the image into an array and preprocess for EfficientNet
        img_array = np.expand_dims(processed_img, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        
        return predicted_class

    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")

