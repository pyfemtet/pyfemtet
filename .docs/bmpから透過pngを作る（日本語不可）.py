import os
import numpy as np
import cv2

here, me = os.path.split(__file__)

path = input('path to bmp: ')
src = cv2.imread(path)
mask = np.all(src[:,:,:] == [255, 255, 255], axis=-1)
dst = cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)
dst[mask,3] = 0
cv2.imwrite(f'{path.replace(".bmp", ".png")}', dst)
