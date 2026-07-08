# ==========================================================
# Contrast Enhancement for Palm Leaf Manuscript Images
# Techniques:
# 1. Histogram Equalization
# 2. CLAHE (Adaptive Histogram Equalization)
# Author: VIVEK S M
# ==========================================================

import cv2
import os
import matplotlib.pyplot as plt

# ==========================================================
# 1. LOAD IMAGE
# ==========================================================

image_path = r".png"

# Check file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"\nImage file does not exist:\n{image_path}")

# Read image
original = cv2.imread(image_path)

# Check image loaded correctly
if original is None:
    raise Exception(f"\nUnable to load image:\n{image_path}")

# Convert BGR → RGB
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# Convert to grayscale
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# ==========================================================
# 2. HISTOGRAM EQUALIZATION
# ==========================================================

hist_equalized = cv2.equalizeHist(gray)

# ==========================================================
# 3. CLAHE
# ==========================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_result = clahe.apply(gray)

# ==========================================================
# 4. DISPLAY RESULTS
# ==========================================================

plt.figure(figsize=(14,8))

plt.subplot(2,2,1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(hist_equalized, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(clahe_result, cmap="gray")
plt.title("CLAHE")
plt.axis("off")

plt.tight_layout()
plt.show()

# ==========================================================
# 5. SAVE OUTPUT
# ==========================================================

output_folder = os.path.join(os.path.dirname(image_path), "Contrast_Enhancement_Output")
os.makedirs(output_folder, exist_ok=True)

hist_path = os.path.join(output_folder, "Histogram_Equalization.png")
clahe_path = os.path.join(output_folder, "CLAHE.png")

cv2.imwrite(hist_path, hist_equalized)
cv2.imwrite(clahe_path, clahe_result)

print("\nContrast enhancement completed successfully.")
print(f"Histogram Equalization saved at:\n{hist_path}")
print(f"CLAHE saved at:\n{clahe_path}")