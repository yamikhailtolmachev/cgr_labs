import numpy as np
import math
from PIL import Image, ImageOps

img_mat = np.zeros((2000, 2000, 3), dtype = np.uint8)
z_b = np.zeros((2000, 2000), dtype = np.float32)

for i in range(2000):
    for j in range(2000):
        z_b[i, j] = 999999999.9

def barycentric(x, y, x0, y0, x1, y1, x2, y2):
    b = []
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)) 
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    b.append(lambda0)
    b.append(lambda1)
    b.append(lambda2)
    return b
    
def triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, color):
    # размер
    ax = 2500
    ay = 2500
    
    # сдвиг картинки
    x0 = (ax * x0) / (z0) + 1000
    y0 = (ay * y0) / (z0) + 1000
    x1 = (ax * x1) / (z1) + 1000
    y1 = (ay * y1) / (z1) + 1000
    x2 = (ax * x2) / (z2) + 1000
    y2 = (ay * y2) / (z2) + 1000
    
    xmin = int(min(x0, x1, x2))
    ymin = int(min(y0, y1, y2))
    
    xmax = int(max(x0, x1, x2)) + 1
    ymax = int(max(y0, y1, y2)) + 1
    
    if (xmin < 0): xmin = 0
    if (ymin < 0): ymin = 0
    
    if (xmax > 2000): xmax = 2000
    if (ymax > 2000): ymax = 2000
    
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            ba = barycentric(x, y, x0, y0, x1, y1, x2, y2)
            if ba[0] > 0 and ba[1] > 0 and ba[2] > 0:
                zk = ba[0] * z0 + ba[1] * z1 + ba[2] * z2
                if zk < z_b[y, x]:
                    img_mat[y, x] = color
                    z_b[y, x] = zk

def file_obj():
    file = open('model_1.obj')
    v = []
    f = []
    
    # поворот модели
    a = np.radians(0)
    b = np.radians(270)
    c = np.radians(0)
    
    r_x = np.array([[1, 0, 0], [0, math.cos(a), math.sin(a)], [0, -math.sin(a), math.cos(a)]])
    r_y = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    r_z = np.array([[math.cos(c), math.sin(c), 0], [-math.sin(c), math.cos(c), 0], [0, 0, 1]])
    r = r_x @ r_y @ r_z
    
    # сдвиг модели
    t = np.array([-0.01, -0.047, 0.17])
    
    for s in file:
        sp = s.split()
        if (sp[0] == 'v'):
            v_a = np.array([float(sp[1]), float(sp[2]), float(sp[3])])
            v_r = r @ v_a + t
            v.append(v_r)
        elif (sp[0] == 'f'):
            spf = []
            for t in sp:
                spf.append(t.split('/')) 
            f.append([spf[1][0], spf[2][0], spf[3][0]])
    
    for i in range(len(f)):
        x0 = float(v[int(f[i][0]) - 1][0])
        y0 = float(v[int(f[i][0]) - 1][1])
        z0 = float(v[int(f[i][0]) - 1][2])
        x1 = float(v[int(f[i][1]) - 1][0])
        y1 = float(v[int(f[i][1]) - 1][1])
        z1 = float(v[int(f[i][1]) - 1][2])
        x2 = float(v[int(f[i][2]) - 1][0])
        y2 = float(v[int(f[i][2]) - 1][1])
        z2 = float(v[int(f[i][2]) - 1][2])
        
        ax, ay, az = x1 - x2, y1 - y2, z1 - z2
        bx, by, bz = x1 - x0, y1 - y0, z1 - z0
        
        # нормали
        n = [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx]
        
        # угол падения света
        alpha = n[2] / ((n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5)
        
        if (alpha < 0):
            triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, [0, 0, -255 * alpha])

# фон
for i in range(2000):
    for j in range(2000):
        img_mat[i, j] = 0, 0, 0
        
file_obj()
    
img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img_16.1.png')