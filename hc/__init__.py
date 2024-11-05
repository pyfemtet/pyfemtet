import os

orca_path = r"C:\Users\mm11592\AppData\Local\Programs\orca"

if orca_path not in os.environ.get('PATH'):
    os.environ['PATH'] = os.environ.get('PATH') + ";" + orca_path


import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)
