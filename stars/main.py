import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import (opening, dilation, closing, erosion)

image = np.load("stars.npy").astype(np.uint8)

struct = np.ones((3, 3))
processed = opening(image, struct)

labled_original = label(image)
labled_processed = label(processed)

print(f"All objects: {labled_original.max()}")
print(f"Rectangles: {labled_processed.max()}")

print(f"Stars: {labled_original.max() - labled_processed.max()}")

stars = np.bitwise_xor(image, processed) # Stars demostration without rectangles

plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(stars)
plt.show()