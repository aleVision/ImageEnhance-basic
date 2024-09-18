import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Title of the app
st.title("Interactive Image Manipulation Web App")

# Sidebar for user inputs
st.sidebar.title("Image Processing Options")
# Upload an image
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Sidebar for user inputs
st.sidebar.title("Batch Image Augmentation Options")
# Upload an image
uploaded_files = st.sidebar.file_uploader("Upload a folder", type=["jpg", "png", "jpeg"])

# Check if an image is uploaded
if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    image = np.array(Image.open(uploaded_file))

    # Display the original image
    st.image(image, caption="Original Image", use_column_width=True)

    # Resize the image
    st.sidebar.subheader("Resize Image")
    new_width = st.sidebar.slider("Width", 50, 800, image.shape[1])
    new_height = st.sidebar.slider("Height", 50, 800, image.shape[0])
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Rotation of the image
    st.sidebar.subheader("Rotate Image")
    rotation_angle = st.sidebar.slider("Rotation Angle", 0, 360, 0)
    center = (resized_image.shape[1] // 2, resized_image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(resized_image, rotation_matrix, (resized_image.shape[1], resized_image.shape[0]))

    # Apply Filters
    st.sidebar.subheader("Apply Filters")
    blur = st.sidebar.checkbox("Gaussian Blur")
    sharpen = st.sidebar.checkbox("Sharpen")
    edges = st.sidebar.checkbox("Edge Detection")
    
    # Apply Gaussian blur
    if blur:
        ksize = st.sidebar.slider("Blur kernel size", 1, 11, 5, step=2)
        rotated_image = cv2.GaussianBlur(rotated_image, (ksize, ksize), 0)

    # Apply sharpening filter
    if sharpen:
        sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        rotated_image = cv2.filter2D(rotated_image, -1, sharpening_kernel)

    # Apply Canny Edge Detection
    if edges:
        low_threshold = st.sidebar.slider("Canny Edge Threshold", 50, 200, 100)
        rotated_image = cv2.Canny(rotated_image, low_threshold, 200)

    # Color space conversion
    st.sidebar.subheader("Color Space Conversion")
    color_space = st.sidebar.selectbox("Convert to Color Space", ["None", "Grayscale", "LAB", "YUV"])

    if color_space == "Grayscale":
        rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    elif color_space == "LAB":
        rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2LAB)
    elif color_space == "YUV":
        rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2YUV)

    # Display the manipulated image
    st.image(rotated_image, caption="Processed Image", use_column_width=True)

    # Save button to download the processed image
    st.sidebar.subheader("Download Processed Image")
    if st.sidebar.button("Save Image"):
        # Save the processed image to a file
        cv2.imwrite("processed_image.png", rotated_image)
        st.sidebar.write("Image saved as 'processed_image.png' in the current directory.")
