import cv2
import numpy as np
import matplotlib.pyplot as plt

def identify_noise(image):
    
    
    unique, counts = np.unique(image, return_counts=True)
    pixel_count = dict(zip(unique, counts))
    
    black_pixels = pixel_count.get(0, 0)
    white_pixels = pixel_count.get(255, 0)
    total_pixels = image.size
    if (black_pixels + white_pixels) / total_pixels > 0.02:  
        return "Salt-and-Pepper"
    
   
    std_dev = np.std(image)
    if std_dev > 20:  
        return "Gaussian"
    
    return "None"

def apply_smoothing(image, noise_type):
    
    if noise_type == "Salt-and-Pepper":
        smoothed_image = cv2.medianBlur(image, 5)  
        filter_used = "Median Filter"
    elif noise_type == "Gaussian":
        smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)  
        filter_used = "Gaussian Filter"
    else:
        smoothed_image = image  
        filter_used = "None"
    return smoothed_image, filter_used


image_path = "DIP_Lab_task_03.png"  
original_image = cv2.imread(image_path)

if original_image is None:
    print("Error: Image could not be loaded. Please check the file path.")
else:
    
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    
    noise_type = identify_noise(grayscale_image)
    print(f"Identified Noise Type: {noise_type}")
    
    
    smoothed_image, filter_used = apply_smoothing(grayscale_image, noise_type)
    print(f"Applied Filter: {filter_used}")
    
    
    median_filtered = cv2.medianBlur(grayscale_image, 5)
    gaussian_filtered = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    
    
    plt.figure(figsize=(15, 8))
    
    
    plt.subplot(2, 2, 1)
    plt.title("Original Image (Grayscale)")
    plt.imshow(grayscale_image, cmap='gray')
    plt.axis('off')
    
    
    plt.subplot(2, 2, 2)
    plt.title("Median Filter (5x5 Kernel)")
    plt.imshow(median_filtered, cmap='gray')
    plt.axis('off')
    
    
    plt.subplot(2, 2, 3)
    plt.title("Gaussian Filter (5x5 Kernel)")
    plt.imshow(gaussian_filtered, cmap='gray')
    plt.axis('off')
    
    
    plt.subplot(2, 2, 4)
    plt.title(f"Smoothed Image ({filter_used})")
    plt.imshow(smoothed_image, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
