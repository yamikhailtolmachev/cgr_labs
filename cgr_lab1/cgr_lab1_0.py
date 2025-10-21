import numpy as np
import math
from PIL import Image

img_mat = np.zeros((200, 200, 3), dtype = np.uint8)

def draw_line(img_mat, x0, y0, x1, y1):
    for x in range (x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round ((1.0 - t) * y0 + t * y1)
        img_mat[y, x] = 100, 23, 145
        
for k in range (13):
    x0, y0 = 100,100
    x1 = int(100 + math.cos(math.pi * k / 13) * 45)
    y1 = int(100 + math.sin(math.pi * k / 13) * 45)
    draw_line(img_mat, x0, y0, x1, y1)
    
img = Image.fromarray(img_mat, mode = 'RGB')
img.save('img_0.png')