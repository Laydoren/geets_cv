import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from skimage.color import rgb2hsv

img = imread('balls_and_rects.png')
hsv = rgb2hsv(img)
h = hsv[:,:,0]

circles = {}
rects = {}

for colour in np.unique(h):
    binary = h == colour
    for region in regionprops(label(binary)):
        if region.area > img.shape[0] * img.shape[1] * 0.2: #background filter
            continue

        if region.eccentricity < 0.5 and region.extent < 0.9 and region.solidity > 0.95:
            circles[colour] = circles.get(colour, 0) + 1

        elif region.extent > 0.85 and region.solidity > 0.95:
            rects[colour] = rects.get(colour, 0) + 1
        else:
            print("Hmm, i dunno")

print(f"All fig: {sum(circles.values()) + sum(rects.values())}")

print(f"\nCircles: {sum(circles.values())}")
for color, num in circles.items():
    print(f"{num}: {color}")

print(f"\nRectangles: {sum(rects.values())}")
for color, num in rects.items():
    print(f"{num}: {color}")


