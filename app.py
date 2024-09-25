import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Title of the app
st.title("Interactive Image Manipulation Web App")

# Sidebar for user inputs
st.sidebar.title("Image Processing Options")

# Add an informative icon and expander for more information
st.sidebar.markdown("ℹ️ **About Image Processing Types**")
with st.sidebar.expander("Learn more about each filter type"):
    st.write("""
    - **Resize**: Adjust the width and height of the image.
    - **Rotate**: Rotate the image by a specified degree angle.
    - **Brightness**: Increase or decrease the brightness of the image by adding/subtracting a constant value to pixel intensities.
    - **Gaussian Blur**: Smoothens the image by reducing image noise using a Gaussian filter.
    - **Sharpen**: Enhances edges in the image, making the details crisper.
    - **Edge Detection**: Detects prominent edges in the image using the Canny algorithm.
    - **Color Space Conversion**: Converts the image to various color spaces like Grayscale, LAB, or YUV.
    """)

# Upload an image
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Check if an image is uploaded
if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    image = np.array(Image.open(uploaded_file))

    # Display the original image
    st.image(image, caption="Original Image", use_column_width=True)

    # Resize the image
    st.sidebar.subheader("Resize Image")
    st.sidebar.markdown("🔧 Resize tool: Adjust the width and height.")
    new_width = st.sidebar.slider("Width", 50, 800, image.shape[1])
    new_height = st.sidebar.slider("Height", 50, 800, image.shape[0])
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Rotation of the image
    st.sidebar.subheader("Rotate Image")
    st.sidebar.markdown("🔄 Rotate tool: Rotate the image by degrees.")
    rotation_angle = st.sidebar.slider("Rotation Angle", 0, 360, 0)
    center = (resized_image.shape[1] // 2, resized_image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(resized_image, rotation_matrix, (resized_image.shape[1], resized_image.shape[0]))

    # Brightness Adjustment
    st.sidebar.subheader("Adjust Brightness")
    st.sidebar.markdown("💡 Brightness tool: Modify the brightness of the image.")
    brightness = st.sidebar.slider("Brightness", -100, 100, 0)  # Slider from -100 to 100
    
    # Adjust brightness by adding or subtracting pixel values
    bright_image = cv2.convertScaleAbs(rotated_image, alpha=1, beta=brightness)

    # Apply Filters
    st.sidebar.subheader("Apply Filters")
    blur = st.sidebar.checkbox("Gaussian Blur")
    st.sidebar.markdown("🌫️ Blur tool: Apply Gaussian Blur to smooth the image.")
    
    sharpen = st.sidebar.checkbox("Sharpen")
    st.sidebar.markdown("🔍 Sharpen tool: Enhance the edges in the image.")

    edges = st.sidebar.checkbox("Edge Detection")
    st.sidebar.markdown("⚡ Edge Detection: Detect the prominent edges in the image.")

    # Apply Gaussian blur
    if blur:
        ksize = st.sidebar.slider("Blur kernel size", 1, 11, 5, step=2)
        bright_image = cv2.GaussianBlur(bright_image, (ksize, ksize), 0)

    # Apply sharpening filter
    if sharpen:
        sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        bright_image = cv2.filter2D(bright_image, -1, sharpening_kernel)

    # Apply Canny Edge Detection
    if edges:
        low_threshold = st.sidebar.slider("Canny Edge Threshold", 50, 200, 100)
        bright_image = cv2.Canny(bright_image, low_threshold, 200)

    # Color space conversion
    st.sidebar.subheader("Color Space Conversion")
    st.sidebar.markdown("🎨 Color Space tool: Convert to Grayscale, LAB, or YUV.")
    color_space = st.sidebar.selectbox("Convert to Color Space", ["None", "Grayscale", "LAB", "YUV"])

    if color_space == "Grayscale":
        bright_image = cv2.cvtColor(bright_image, cv2.COLOR_BGR2GRAY)
    elif color_space == "LAB":
        bright_image = cv2.cvtColor(bright_image, cv2.COLOR_BGR2LAB)
    elif color_space == "YUV":
        bright_image = cv2.cvtColor(bright_image, cv2.COLOR_BGR2YUV)
    elif color_space == "HSV":
        bright_image = cv2.cvtColor(bright_image, cv2.COLOR_BGR2HSV)

    # Display the manipulated image
    st.image(bright_image, caption="Processed Image", use_column_width=True)

    # Save button to download the processed image
    st.sidebar.subheader("Download Processed Image")
    if st.sidebar.button("Save Image"):
        # Save the processed image to a file
        cv2.imwrite("processed_image.png", bright_image)
        st.sidebar.write("Image saved as 'processed_image.png' in the current directory.")
