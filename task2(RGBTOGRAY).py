#Task to RGBto Gray


import cv2
import numpy as np
import matplotlib.pyplot as plt

imgcat = cv2.imread('DIP_Lab_task_02.jpg')

def convert_to_grayscale(image):
    
    height, width, _ = image.shape
    
    gray_image = np.empty((height, width), dtype=np.float32)
    
    for row in range(height):
        for col in range(width):
            blue, green, red = image[row, col]
            gray_image[row, col] = int(0.3 * red + 0.59 * green + 0.1 * blue)
    
    return gray_image

grayscale_img = convert_to_grayscale(imgcat)

plt.imshow(grayscale_img, cmap='gray')
plt.axis('off')
plt.title('Grayscale Image of the task2cat')
plt.show()