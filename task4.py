import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2 
def gamma_correction(image, gamma, scale=1.0):
    
    normalized_image = image / 255.0
    
    corrected_image = scale * (normalized_image ** gamma)
    
    corrected_image = np.clip(corrected_image, 0, 1) * 255
    return corrected_image.astype(np.uint8)

image4 = io.imread('DIP_Lab_task_04.jpg')
plt.figure(figsize=(15, 8))
plt.subplot(1, 5, 2)
plt.imshow(gamma_correction(image4, 3))
plt.title('Gamma = 3')

plt.subplot(1, 5, 3)
plt.imshow(gamma_correction(image4, 5))
plt.title('Gamma = 5')

#scalling
scaling_factor = 0.4  
darkened_image = cv2.convertScaleAbs(image4, alpha=scaling_factor, beta=0)
plt.figure(figsize=(6, 6))
plt.title('brightnexx reduced Image using scalling')
plt.imshow(darkened_image)
plt.axis('off')

plt.show()