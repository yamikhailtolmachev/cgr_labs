import numpy as np
import math
from PIL import Image, ImageOps

# img_mat = np.zeros((200, 200), dtype = np.uint8)
img_mat = np.zeros((2000, 2000, 3), dtype = np.uint8)

# for i in range (200):
#     for j in range (200):
#         img_mat[i, j] = 0, 0, 0

# def dotted_line(img_mat, x0, y0, x1, y1, count, color):
#     step = 1.0 / count
#     for t in np.arange (0, 1, step):
#         x = round ((1.0 - t) * x0 + t * x1)
#         y = round ((1.0 - t) * y0 + t * y1)
#         img_mat[y, x] = color

# def dotted_line_v2(img_mat, x0, y0, x1, y1, color):
#     count = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
#     step = 1.0 / count
#     for t in np.arange (0, 1, step):
#         x = round ((1.0 - t) * x0 + t * x1)
#         y = round ((1.0 - t) * y0 + t * y1)
#         img_mat[y, x] = color

# def x_loop_line(img_mat, x0, y0, x1, y1, color):
#     for x in range (x0, x1):
#         t = (x - x0) / (x1 - x0)
#         y = round ((1.0 - t) * y0 + t * y1)
#         img_mat[y, x] = color

# def x_loop_line_hotfix_1(img_mat, x0, y0, x1, y1, color):
#     if (x0 > x1):
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
#     for x in range (x0, x1):
#         t = (x - x0) / (x1 - x0)
#         y = round ((1.0 - t) * y0 + t * y1)
#         img_mat[y, x] = color

# def x_loop_line_hotfix_2(img_mat, x0, y0, x1, y1, color):
#     if (x0 > x1):
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0

#     xchange = False
#     if (abs(x0 - x1) < abs(y0 - y1)):
#         x0, y0 = y0, x0
#         x1, y1 = y1, x1
#         xchange = True
    
#     for x in range(x0, x1):
#         t = (x - x0) / (x1 - x0)
#         y = round ((1.0 - t) * y0 + t * y1)
#         if (xchange):
#             img_mat[x, y] = color
#         else:
#             img_mat[y, x] = color

# def x_loop_line_v2(img_mat, x0, y0, x1, y1, color):
#     xchange = False
#     if (abs(x0 - x1) < abs(y0 - y1)):
#         x0, y0 = y0, x0
#         x1, y1 = y1, x1
#         xchange = True
    
#     if (x0 > x1):
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
    
#     for x in range(x0, x1):
#         t = (x - x0) / (x1 - x0)
#         y = round ((1.0 - t) * y0 + t * y1)
#         if (xchange):
#             img_mat[x, y] = color
#         else:
#             img_mat[y, x] = color

# def x_loop_line_v2_no_y_calc(img_mat, x0, y0, x1, y1, color):
#     xchange = False
#     if (abs(x0 - x1) < abs(y0 - y1)):
#         x0, y0 = y0, x0
#         x1, y1 = y1, x1
#         xchange = True
    
#     if (x0 > x1):
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
    
#     y = y0
#     dy = abs(y1 - y0) / (x1 - x0)
#     derror = 0.0
#     y_update = 1 if y1 > y0 else -1
    
#     for x in range(x0, x1):
#         if (xchange):
#             img_mat[x, y] = color
#         else:
#             img_mat[y, x] = color
        
#         derror += dy
#         if (derror > 0.5):
#             derror -= 1.0
#             y += y_update
    
# def x_loop_line_v2_no_y_calc_v2_for_some_unknown_reason(img_mat, x0, y0, x1, y1, color):
#     xchange = False
#     if (abs(x0 - x1) < abs(y0 - y1)):
#         x0, y0 = y0, x0
#         x1, y1 = y1, x1
#         xchange = True
        
#     if (x0 > x1):
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
    
#     y = y0
#     dy = 2.0 * (x1 - x0) * abs(y1 - y0) / (x1 - x0)
#     derror = 0.0
#     y_update = 1 if y1 > y0 else -1
    
#     for x in range(x0, x1):
#         if (xchange):
#             img_mat[x, y] = color
#         else:
#             img_mat[y, x] = color
        
#         derror += dy
#         if (derror > 2.0 * (x1 - x0) * 0.5):
#             derror -= 2.0 * (x1 - x0) * 1.0
#             y += y_update

def bresenham_line(img_mat, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x1 - x0) < abs(y1 - y0)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
        
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    
    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    
    for x in range(x0, x1):
        if (xchange):
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color
        
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2 * (x1 - x0)
            y += y_update

# def task_3():
#     file = open('model_1.obj')
#     v = []
#     for s in file:
#         sp = s.split()
#         if (sp[0] == 'v'):
#             v.append([float(sp[1]), float(sp[2]), float(sp[3])])

#     print(v)

# def task_4():
#     file = open('model_1.obj')
#     v = []
#     for s in file:
#         sp = s.split()
#         if (sp[0] == 'v'):
#             v.append([float(sp[1]), float(sp[2]), float(sp[3])])
        
#     for i in range(len(v)):
#         x = int(8000 * v[i][0] + 1000)
#         y = int(8000 * v[i][1] + 1000)
#         img_mat[y, x] = 255, 255, 255

# def task_5():
#     file = open('model_1.obj')
#     f = []
#     for s in file:
#         sp = s.split()
#         if (sp[0] == 'f'):
#             spf = []
#             for t in sp:
#                 spf.append(t.split('/'))
#             f.append([spf[1][0], spf[2][0], spf[3][0]])
            
#     print(f)

def file_obj():
    file = open('model_1.obj')
    v = []
    f = []
    
    for s in file:
        sp = s.split()
        if (sp[0] == 'v'):
            v.append([sp[1], sp[2], sp[3]])
        elif (sp[0] == 'f'):
            spf = []
            for t in sp:
                spf.append(t.split('/'))
            f.append([spf[1][0], spf[2][0], spf[3][0]])

    for i in range(len(f)):
        x0 = int(8000 * float(v[int(f[i][0]) - 1][0]) + 1000)
        y0 = int(8000 * float(v[int(f[i][0]) - 1][1]) + 1000)
        x1 = int(8000 * float(v[int(f[i][1]) - 1][0]) + 1000)
        y1 = int(8000 * float(v[int(f[i][1]) - 1][1]) + 1000)
        x2 = int(8000 * float(v[int(f[i][2]) - 1][0]) + 1000)
        y2 = int(8000 * float(v[int(f[i][2]) - 1][1]) + 1000)
        
        bresenham_line(img_mat, x0, y0, x1, y1, [75, 30, 175])
        bresenham_line(img_mat, x1, y1, x2, y2, [75, 30, 175])
        bresenham_line(img_mat, x0, y0, x2, y2, [75, 30, 175])

for i in range(2000):
    for j in range(2000):
        img_mat[i, j] = 200, 255 - (i * 255 / 2000 + j * 255 / 2000) / 5, 255

# for k in range (13):
#     x0, y0 = 100, 100
#     x1 = int(100 + math.cos(math.pi * 2 * k / 13) * 95)
#     y1 = int(100 + math.sin(math.pi * 2 * k / 13) * 95)
#     dotted_line(img_mat, x0, y0, x1, y1, 100, 255)
#     dotted_line_v2(img_mat, x0, y0, x1, y1, 255)
#     x_loop_line(img_mat, x0, y0, x1, y1, 255)
#     x_loop_line_hotfix_1(img_mat, x0, y0, x1, y1, 255)
#     x_loop_line_hotfix_2(img_mat, x0, y0, x1, y1, 255)
#     x_loop_line_v2(img_mat, x0, y0, x1, y1, 255)
#     x_loop_line_v2_no_y_calc(img_mat, x0, y0, x1, y1, 255)
#     x_loop_line_v2_no_y_calc_v2_for_some_unknown_reason(img_mat, x0, y0, x1, y1, 255)
#     bresenham_line(img_mat, x0, y0, x1, y1, 255)

# task_3()
# task_4()
# task_5()
file_obj()
    
# img = Image.fromarray(img_mat, mode = 'L')
img = Image.fromarray(img_mat, mode = 'RGB')
img = ImageOps.flip(img)
img.save('img_6.png')