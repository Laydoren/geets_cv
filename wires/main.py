import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.morphology import (opening, dilation, closing, erosion)

image = np.load("wires/wires6.npy")

struct = np.ones((3, 1))
processed = opening(image, struct)
labled = label(image)

for n in range(1, labled.max()+1):
    parts = label(labled == n)
    parts_processed = label(opening(parts, struct)).max()
    if parts_processed != 0:
        print(f"Wire = {n}, parts = {parts_processed}")

plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(processed)
plt.show()
