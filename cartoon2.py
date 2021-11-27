import cv2
import numpy as np
import matplotlib.image as img
from matplotlib import pyplot as plt

#reading the image
img = cv2.imread("Cartoon\ironman.png")

#Create Edge Mask
def edge_mask(img, line_size, blur_value):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_blur = cv2.medianBlur(gray, blur_value)
  edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
  return edges

line_size = 7
blur_value = 7

edges = edge_mask(img, line_size, blur_value)

filename = 'edges.jpg'
# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, edges)

#colour quantization
#k value determines the number of colours in the image
total_color = 8
k=total_color

# Transform the image
data = np.float32(img).reshape((-1, 3))

# Determine criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

# Implementing K-Means
ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
result = center[label.flatten()]
result = result.reshape(img.shape)

filename = 'colour.jpg'
# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, result)

blurred = cv2.bilateralFilter(result, d=10, sigmaColor=250,sigmaSpace=250)

#saving the image
filename = 'blurred.jpg'
# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, blurred)

#blurred and edges
cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
filename = 'cartoon.jpg'
# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, cartoon)

#Colour Quantization
def ColourQuantization(image, K=9):
    Z = image.reshape((-1, 3)) 
    Z = np.float32(Z) 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    compactness, label, center = cv2.kmeans(Z, K, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    return res2
#to get countours
def Countours(image):
    contoured_image = image
    gray = cv2.cvtColor(contoured_image, cv2.COLOR_BGR2GRAY) 
    edged = cv2.Canny(gray, 200, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
    cv2.drawContours(contoured_image, contours, contourIdx=-1, color=6, thickness=1)
    return contoured_image

image = cv2.imread("Cartoon\ironman.png")
coloured = ColourQuantization(image)
contoured = Countours(coloured)
final_image = contoured

filename = 'cartoon_final.jpg'
# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, final_image)


