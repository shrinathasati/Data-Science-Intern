# Data Science Intern LGMVIP FEB 2023
# Task-2 "Image to Pencil Sketch with Python"

# Importing important libraries...
import cv2
from google.colab.patches import cv2_imshow

# Uploading and Reading the image file...
image=cv2.imread("/images.jpg")
cv2_imshow(image)

# Converting image into gray format...
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2_imshow(gray_image)

# Inverted the image...
inverted_image=255-gray_image
cv2_imshow(inverted_image)

# Finally converting our image into sketching...
blurred=cv2.GaussianBlur(inverted_image,(21,21),0)
inverted_blur=255-blurred
pencil_sketch=cv2.divide(gray_image,inverted_blur,scale=256.0)
cv2_imshow(pencil_sketch)

# Finally our result are as:-
print("orignal image is:-")
cv2_imshow(image)
print("sketching is:-")
cv2_imshow(pencil_sketch)
