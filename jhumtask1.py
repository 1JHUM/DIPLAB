# task1 image dimention,range,frequency 
import cv2
import numpy as np
import matplotlib.pyplot as plt

img1_path = 'DIP_Lab_task_01.png'
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

dimensions = img1.shape
intensity_values, frequency_distribution = np.unique(img1, return_counts=True)
intensity_range = (img1.min(), img1.max())

print("Image Dimensions:", dimensions)
print("Intensity Range:", intensity_range)
print("Frequency Distribution of Intensities:", dict(zip(intensity_values, frequency_distribution)))

#  histogram of intensity values
plt.figure(figsize=(10, 6))
plt.bar(intensity_values, frequency_distribution, color='blue')
plt.title('Histogram of Intensity Values')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
