
# import os
# here, me = os.path.split(__file__)
# import sys
# sys.path.append(here)
# mother_path = os.path.abspath(os.path.join(here, '..'))
# sys.path.append(mother_path)

import numpy as np
import pandas as pd

# plot ライブラリ
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 日付
import matplotlib.dates as mdates

# 日本語
import japanize_matplotlib

# dpi
import ctypes
scaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0)

# フォント設定
from matplotlib import rcParams
rcParams["font.size"] = 10.5
rcParams["figure.dpi"] = scaleFactor
# rcParams['font.family'] = 'MS Gothic'

# スタイル
plt.style.use("ggplot")

# 3D
from mpl_toolkits.mplot3d import Axes3D

# # clipboard
# import io
# from PyQt5.QtGui import QImage
# from PyQt5.QtWidgets import QApplication

# 型判定のためインポート
from matplotlib.collections import PathCollection


