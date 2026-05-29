import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

save_path = Path(__file__).parent

def count_holes(region):
    shape = region.image.shape
    new_shape = np.zeros((shape[0] + 2, shape[1] + 2))
    new_shape[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_shape)
    labeled = label(new_image)
    return np.max(labeled) - 1

def count_lines(region):
    shape = region.image.shape
    image = region.image
    vlines = (np.sum(image, 0) / shape[0] == 1).sum()
    hlines = (np.sum(image, 1) / shape[1] == 1).sum()
    return vlines, hlines

def simmetry(region, transpose=False):
    image = region.image
    if transpose:
        image = image.T
    shape = image.shape

    top = image[:shape[0] // 2]
    if shape[0] % 2 != 0:
        bottom = image[shape[0] // 2 + 1:]
    else:
        bottom = image[shape[0] // 2:]

    bottom = bottom[::-1]
    result = bottom == top
    return result.sum() / result.size

def classificator(region):
    holes = count_holes(region)
    if holes == 2: #B, 8
        v, _ = count_lines(region)
        v /= region.image.shape[1]
        if v > 0.2:
            return "B"
        else:
            return "8"
    elif holes == 1: #A, O, P, D
        # print(simmetry(region),simmetry(region, transpose=True))
        hor = simmetry(region)
        ver = simmetry(region, transpose=True)
        if hor > 0.8 and ver > 0.8:
            return "O"
        elif hor > 0.9 and ver > 0.6:
            return "D"
        elif hor > 0.3 and ver > 0.8:
            return "A"
        else:
            return "P"

    elif holes == 0: #1, W, X, *, -. /
        hor = simmetry(region)
        ver = simmetry(region, transpose=True)


        if hor == 1 and ver == 1:
            return "-"
        elif hor > 0.7 and ver > 0.7:
            # print(region.label, region.eccentricity)
            if region.eccentricity > 0.85:
                return "1"
            else:
                return "X"
        elif hor > 0.3 and ver > 0.8:
            # print(region.label, region.eccentricity)
            if region.eccentricity > 0.55:
                return "W"
            else:
                return "*"
        else:
            # print(region.label, hor, ver)
            return "/"


    return "?"


image = imread('symbols.png')[:,:,:-1]
abinary = image.mean(2) > 0
alabeled = label(abinary)
print(np.max(alabeled))

aprops = regionprops(alabeled)

result = {}
image_path = save_path / "out_3"
image_path.mkdir(exist_ok=True)

# plt.ion()
plt.figure(figsize=(5,7))

for region in aprops:
    symbol = classificator(region)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] += 1
    plt.cla()
    plt.title(f"Class: {symbol}")
    plt.imshow(region.image)
    plt.savefig(image_path / f"{region.label}.png")
print(result)



plt.imshow(abinary)
plt.show()