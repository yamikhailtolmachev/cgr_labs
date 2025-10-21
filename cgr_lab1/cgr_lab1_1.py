import numpy as np
from PIL import Image

H = int(input())
W = int(input())

# img_mat = np.zeros((H, W), dtype = np.uint8)
img_mat = np.zeros((H, W, 3), dtype = np.uint8)

for x in range (0, W):
    for y in range (0, H):
        # img_mat[y, x] = 255
        # img_mat[y, x] = 255, 0, 0
        img_mat[y, x] = (y * 255 // H), ((x + y) * 255 // (H + W)), (x * 255 // W)

# img = Image.fromarray(img_mat, mode = 'L')
img = Image.fromarray(img_mat, mode = 'RGB')
img.save('img_1.png')