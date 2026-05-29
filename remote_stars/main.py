import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from scipy import ndimage
import socket

host = "84.237.21.36"
port = 5152


def recvall(sock, nbytes):
    data = bytearray()
    while len(data) < nbytes:
        packet = sock.recv(nbytes - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


plt.ion()
plt.figure()
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((host, port))
    i = 0
    beat = b"nope"
    while beat != b"yep":
        sock.send(b"get")
        bts = recvall(sock, 40002)

        im1 = np.frombuffer(bts[2:40002], dtype=np.uint8)
        im1 = im1.reshape(bts[0], bts[1])

        binary = im1 > 10
        labeled = label(binary)

        centroids = ndimage.center_of_mass(binary, labeled, range(1, labeled.max() + 1))

        print(centroids)

        p1 = np.array(centroids[0])
        p2 = np.array(centroids[-1])

        print(p1)
        print(p2)

        distance = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        i = i+1
        print(f"{i}: {distance:.1f}")

        sock.send(f"{distance:.1f}".encode())
        print(sock.recv(100))

        plt.clf()
        plt.imshow(labeled)
        plt.pause(0.1)

        sock.send(b"beat")
        beat = sock.recv(100)