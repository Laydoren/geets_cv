import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.measure import label
import os

folder = "out"  # name of your folder in project

images = []
for i in range(100):
    img = np.load(os.path.join(folder, f"h_{i}.npy"))
    images.append(img)

def get_centroids(img):
    labeled = label(img)
    num = labeled.max()
    centroids = ndimage.center_of_mass(img, labeled, range(1, num + 1))

    return centroids

trajectories = []
for i, img in enumerate(images):

    centroids = get_centroids(img)

    if i == 0:
        trajectories = [[c] for c in centroids]

    else:
        for traj in trajectories:
            last = traj[-1]

            dists = [((c[0] - last[0]) ** 2 + (c[1] - last[1]) ** 2) ** 0.5 for c in centroids]
            idx = np.argmin(dists)

            traj.append(centroids[idx])
            centroids.pop(idx)


plt.figure()

for traj in trajectories:
    ys, xs = zip(*traj)
    plt.plot(xs, ys, marker='o')

plt.grid()
plt.show()