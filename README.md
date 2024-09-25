# Basic Image Processing Manipulations - Learning concepts

This project aims to be an interactive way to get first contact with basic and most common image processing using Opencv.
The objective is to show first steps that include:

* Load an image.
* Resize and rotate the image.
* Apply basic filters like blurring, sharpening, and edge detection.
* Perform histogram equalization to improve contrast.
* Convert the image to different color spaces (LAB, YUV).

## 1. Load an Image
Using OpenCV to load the image and display it.
```
import cv2

# Load an image from a file
image = cv2.imread('artwork.jpg')

# Display the image
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 2. Image Resizing and Rotation
Allow users to resize and rotate their images
```
# Resize the image to a specific width and height
resized_image = cv2.resize(image, (300, 300))

# Rotate the image by 90 degrees clockwise
rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Display the resized and rotated images
cv2.imshow('Resized Image', resized_image)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 3. Basic Filters
Allow users to enhance their images with simple filters like blurring, sharpening, and edge detection
```
# Apply a Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Apply sharpening using a kernel
sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)

# Detect edges using Canny edge detector
edges = cv2.Canny(image, 100, 200)

# Display the processed images
cv2.imshow('Blurred Image', blurred_image)
cv2.imshow('Sharpened Image', sharpened_image)
cv2.imshow('Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 4. Perform Histogram Equalization
This technique can enhance the contrast of an image, particularly in cases where the lighting is not ideal
```
# Convert the image to grayscale for histogram equalization
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization
equalized_image = cv2.equalizeHist(gray_image)

# Display the original and equalized images
cv2.imshow('Original Grayscale Image', gray_image)
cv2.imshow('Histogram Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 5. Color Space Conversion
Allow the user to manipulate the image by converting it to different color spaces like LAB or YUV for artistic effects.
```
# Convert the image from RGB to LAB color space
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Convert the image from RGB to YUV color space
yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

# Display the images in different color spaces
cv2.imshow('LAB Color Space', lab_image)
cv2.imshow('YUV Color Space', yuv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Why it’s Useful:
In the real world, tools like these are vital in applications for:

* Improving image quality for publication (media, digital art).
* Preprocessing images for computer vision tasks (recognition, segmentation).
* Enhancing images taken under suboptimal lighting conditions for personal use or professional photography.

This project, though basic, forms the building blocks for more complex image manipulation tasks like object detection, image segmentation, and style transfer, making it highly relevant in both industry and academia.

