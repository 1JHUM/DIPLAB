import cv2
import matplotlib.pyplot as plt

imgapple = cv2.imread('DIP_Lab_task_05.jpg')

imgapple_gray = cv2.cvtColor(imgapple, cv2.COLOR_BGR2GRAY)

#  threshold value
threshold = 120

rows, cols = imgapple_gray.shape

# Apply thresholding
for i in range(rows):
    for j in range(cols):
        if imgapple_gray[i, j] > threshold:
            imgapple_gray[i, j] = 255
        else:
            imgapple_gray[i, j] = 0

# Display the image
plt.imshow(imgapple_gray, cmap='gray')
plt.axis('off')
plt.title('Segmented Image')
plt.show()