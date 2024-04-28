import os
import torch
import torch.nn as nn

import torch.utils
from torch.utils.data import DataLoader, Dataset

from torchvision.utils import save_image, make_grid
from tqdm import tqdm

import numpy as np

from torchvision import transforms

from UNet import *



# ⠀⢀⠀⢀⣀⣠⣤⣤⣤⣤⣤⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣠⣠⣤⣤⣤⣤⣀⠲⢦⣄⡀⠀⠀
# ⡶⢟⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦⡀⠀⠀⠀⠀⠀⠰⣷⣷⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣬⡛⢷⣔
# ⣾⡿⠟⠋⠉⠁⠀⡀⠀⠀⠀⠀⠈⠉⠉⠙⠛⢻⠛⠛⠋⠀⠀⠀⠀⠀⠀⠀⠈⠙⢛⣛⣛⣛⣛⣉⢉⣉⡀⠀⠀⠀⠀⠀⠈⠉⠛⢿⣷⣝
# ⠃⠀⠀⠀⠀⠀⠀⣛⣛⣛⣛⣛⣛⢛⡛⠛⠛⠛⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣌⠛⠛⢛⣛⣛⣛⣛⣛⣛⣛⣓⣀⠀⠀⠀⠀⠀⠈⢻
# ⠀⠀⠀⢀⣤⡾⠛⢻⣿⣿⣿⡿⣿⡟⢻⣿⠳⠆⠘⣿⣦⠀⠀⠀⠀⠀⠀⠀⣰⣿⠁⠐⠛⣿⡟⢻⣿⣿⣿⣿⢿⣟⠛⠻⣦⣀⠀⠀⠀⠀
# ⠀⠀⢴⠿⣧⣄⣀⣘⣿⣿⣿⣿⣿⡿⣀⡙⢷⠀⢀⡿⠁⠀⠀⠀⠀⠀⠀⠀⠈⢻⡖⠀⣾⣋⣀⣺⣿⣿⣿⣿⣿⣏⣀⣤⣴⠿⢷⠀⠀⠀
# ⠀⠀⠀⠀⠈⠉⠉⠉⠉⠉⠉⠙⠉⠉⠉⠉⠛⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠋⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠁⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠆⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⠉⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡿⠁⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⠆⠀⠀⢀⣿⠁⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣶⠟⠁⠀⠀⠀⣾⠇⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣤⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⣀⣀⣤⣤⣴⣶⣾⠿⠛⠋⠀⠀⠀⠀⠀⢸⡟⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠟⠛⠛⠛⠛⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠇⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠋⠀⠀⠀




# <3
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣤⡤⠴⠶⠶⠖⠒⠛⠛⠛⠛⠛⠛⠛⠋⠓⠲⠶⠶⠦⢤⣤⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣤⠴⠞⠛⠋⠉⠁⠀⠀⢀⣀⣀⣀⣀⣀⠠⠄⠤⠄⠤⠄⣀⣀⣀⣀⡀⠀⠀⠀⠀⠉⠙⠛⠲⠦⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣤⣤⣴⣾⣭⣕⣀⣀⠠⠤⠔⠒⠊⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠑⠒⠂⠤⢄⣀⣀⣩⣽⣷⣦⣤⣤⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣶⣿⣿⠿⠟⠛⠉⡉⠍⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀  ⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠩⢉⡉⠛⠻⠿⣿⣿⣶⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⣿⡿⠋⠁⢀⠠⠂⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀     ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠐⠤⡀⠈⠙⢿⣿⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⢞⣿⡿⠋⢀⠤⠊⣁⠤⠐⣒⣀⣠⣀⣒⣒⠠⠄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⠄⣒⣂⣠⣀⣀⣒⠂⠤⣀⠑⠢⡀⠙⢿⣿⠳⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡴⠋⠁⣾⡟⡡⠒⢁⠔⣪⣴⣾⡿⠿⠿⠟⠻⠿⠿⢿⣷⣯⣕⠦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣮⣷⣾⡿⠿⠿⠟⠻⠿⠿⢿⣷⣦⣍⠢⣈⠒⢌⣻⣇⡈⠙⢦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⠋⠀⡠⠊⠘⠋⢀⡔⣥⣾⠿⠋⠁⠀⠀⢀⣀⣀⣀⡀⠀⠈⠙⠻⣷⣔⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣶⣿⠟⠋⠁⠀⢀⣀⣀⣀⡀⠀⠀⠈⠙⠻⣷⣜⠧⡀⠙⠃⠑⢤⠀⠙⠷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡞⠃⠀⡠⠊⠀⠀⠀⢠⣮⣾⡿⠃⠀⠀⣀⣴⣿⠿⡻⢟⡻⣿⣷⣦⡀⠀⠈⠻⣿⣂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⡟⠋⠀⢀⣴⣿⡿⢿⡛⢿⠿⣟⣦⣄⠀⠀⠈⢻⣷⡘⡄⠀⠀⠀⠑⢄⠀⠁⢷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡾⠋⠀⢀⠜⠀⠀⠀⠀⢀⢃⣾⠏⠀⠀⢠⣴⡿⠋⠉⠉⠳⡌⠰⢱⢺⣽⣿⣆⠀⠀⠹⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⠏⠀⠀⣠⣿⠿⠛⠚⢥⡈⠂⡑⢎⢯⣿⣧⡀⠀⠀⠹⣷⠸⡄⠀⠀⠀⠀⠣⡀⠀⠙⢷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⠟⠀⠀⢀⠎⠀⠀⠀⠀⠀⢸⢻⡏⠀⠀⠀⣾⣿⠁⠀⠀⠀⠀⢹⣿⣧⣞⣞⡷⣿⡆⠀⠀⢹⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⡏⠀⠀⢠⣿⠇⠀⠀⠀⠀⢻⣿⣷⣮⢲⣽⣻⣷⡀⠀⠀⢹⡇⢇⠀⠀⠀⠀⠀⠑⡀⠀⠀⠹⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡾⠁⠂⠀⢀⠎⠀⠀⠀⠀⠀⠀⠐⣿⠃⠀⠀⣸⣿⢿⡄⠀⠀⠀⢀⣿⣿⣿⣿⣾⣻⣽⣷⠀⠀⠘⣿⠆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⠁⠀⠀⣾⣿⡄⠀⠀⠀⠀⢸⣿⣿⣿⣿⣾⣳⣿⡇⠀⠀⠘⣿⠘⠀⠀⠀⠀⠀⠀⠘⡀⠀⠀⠈⢷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⣠⠟⠀⠀⠀⠀⡌⠀⠀⠀⠀⠀⠀⠀⢠⢻⡄⠀⠀⣾⣿⣻⣿⣶⣶⣶⣿⣿⣿⡿⠛⢻⣷⣻⡿⠀⠀⢠⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⠀⠀⠀⣿⣿⢿⣤⣀⣀⣴⣿⣿⣿⡟⠀⠈⣿⣞⣿⠀⠀⢠⡿⠄⠀⠀⠀⠀⠀⠀⠀⢡⠀⠀⠀⠀⠻⣆⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⣴⠋⡀⠀⠀⠀⠠⠁⠀⠀⠀⠀⢀⣀⣤⠤⠿⠗⠚⠛⠋⠀⠉⠈⠉⠉⠉⠉⠁⠉⠙⠓⠒⠋⠿⠧⣤⣀⢰⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣇⢀⣤⡼⠿⠝⠛⠛⠋⠁⠉⠉⠉⠉⠉⠉⠉⠈⠘⠛⠓⠚⠷⠤⣤⣀⡀⠀⠀⠈⠀⠀⠀⠀⠀⠹⣦⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⣼⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠞⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠂⠬⢙⠶⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠴⠋⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠳⣄⠀⠀⠀⠀⠀⠀⠀     ⠀⠀⠘⣧⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⣼⠃⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⢄⣢⢒⡴⣒⢦⠲⣄⢢⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡄⡰⢄⡲⣔⠮⣔⡢⡔⡠⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀    ⠀⠀⠀⠀⠀⠘⣧⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⣸⠃⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠈⠎⠔⢫⠜⢭⠚⡝⢌⠣⠈⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠈⠱⣉⠳⣩⠛⡬⢓⠱⠁⠃⠀⠀⠀⠀⠀⠀⠀   ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣧⠀⠀⠀⠀
# ⠀⠀⠀⢰⡏⠐⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀       ⠀⠀⠀⠀⠀⠀⠀⠀⢹⡆⠀⠀⠀
# ⠀⠀⢀⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡄⠀⠀
# ⠀⠀⣼⠁⠈⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀     ⠀⠀⠀⠀⠀⠀⠀⠈⣧⠀⠀
# ⠀⢠⡏⠐⠀⠀⠀⠀⢀⠔⠁⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣠⣤⣤⣶⡶⠶⠶⠟⠛⠛⠛⠛⠉⠉⠉⠉⠉⠉⠉⢹⠏⠉⠉⠉⠉⠉⠉⠉⠙⠛⠛⠛⠛⠶⠶⢶⣶⣤⣤⣄⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠈⠒⢄⡀⠀⠀⠀⠀⠀⢹⡆⠀
# ⠀⣼⠁⠈⠀⠀⠀⡐⠁⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⣀⣤⡤⣶⣶⣿⣿⣿⣿⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣿⣿⣿⣾⣿⣽⣶⣲⣤⣄⣀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀ ⠀⠀⠀⠀⡈⠢⡀⠀ ⠀⠀⠘⣷⠀
# ⢀⡟⠐⠀⠀⠀⡜⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⣀⣤⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣿⣽⣗⣦⣤⣀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠑⠄⠑⡄⠀⠀⠀⢿⡄
# ⢸⡇⠐⠀⠀⢰⠃⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⣠⣴⣿⣽⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣯⣟⡶⣄⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠸⡀⠀⢸⡇
# ⠸⡅⠂⠀⠀⠘⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣮⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣷⡽⣄⠀⠀⠀⠀⠀ ⠀⠀⠀⠇⠀⠈⣇
# ⣿⠀⡀⠀⠀⠀⠀⠀⠀⡀⣀⣠⣴⠞⢻⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡷⡏⠳⣦⣤⣀⣀⠀⠀ ⠀⠀⠀⣿
# ⣿⠀⠀⠀⠀⠀⠀⠀⠀⠋⠉⠁⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⣻⣽⣾⣿⣿⣿⣾⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣯⣿⣿⣿⢿⣿⣿⣿⣿⣿⣿⣿⢳⠁⠀⠀⠈⠉⠙⠀ ⠀  ⠀⠀⠀⣿
# ⣿⠀⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣿⣿⣿⣿⣿⣿⣯⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣻⣽⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⣿⡿⡝⠀⠀⠀⠀⠀⠀⠀ ⠀ ⠀ ⠀⠀⠀⣿
# ⣿⠀⢂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣿⣿⣿⣷⣿⣿⣿⣿⣿⣿⣿⢿⣿⣻⣽⣾⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⢷⠃⠀⠀⠀⠀⠀⠀ ⠀  ⠀⠀⡀⠀⣿
# ⢸⡀⢺⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢳⢿⣿⣿⣿⢿⣟⣿⣽⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⡷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⣿⣿⣿⣿⣾⣿⣿⣿⣽⣿⣿⣿⣷⣿⣿⣿⣷⡿⡞⠀⠀⠀⠀⠀⠀ ⠀ ⠀ ⠀⠀⡇⢈⡏
# ⢸⡇⢈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⣿⡿⣟⣿⣿⣿⡧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⣿⣿⣿⡿⣿⣿⣯⣿⣿⣿⣿⣽⣿⣿⣟⣿⣿⣳⠁⠀⠀⠀⠀⠀⠀   ⠀⠀⢠⠃⢸⡇
# ⠈⣷⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣿⣿⣿⢿⣿⣻⣯⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣗⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣟⣿⣿⣿⡿⣿⣿⣏⠇⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⡜⠀⣾⠁
# ⠀⢻⡀⡘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣻⣿⣿⣟⣿⣿⣿⣷⣵⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣶⣿⣿⣿⣿⣿⣿⣷⣿⣿⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⡿⡞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠇⢠⡟⠀
# ⠀⠘⣇⢀⢣⠀⡀⠐⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⡿⣟⣿⣿⣯⣿⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣾⣵⣦⣤⣀⣀⠀⠀⠀⠀⠀⠀⢸⡃⠀⠀⠀⠀⠀⠀⣀⣀⣤⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣾⣿⣿⣿⣯⣿⣿⣿⣿⢿⣻⣷⣿⡿⡹⠀⠀⠀⠀⠀⠀⠀⠀⠀⠆⡜⠀⣸⠃⠀
# ⠀⠀⢿⡄⠈⡆⠰⠀⢡⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣯⣿⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣿⣿⣿⣷⣦⣶⣾⣷⣶⣴⣮⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿⣷⣿⣿⣿⣿⣽⣿⣿⣿⣿⣾⣿⣿⣿⣿⡻⠁⠀⠀⠀⠀⡌⠀⠌⢠⠃⢠⡟⠀⠀
# ⠀⠀⠈⣷⠀⡰⡀⠂⠀⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢾⣿⣿⣿⣿⣿⣽⣿⣿⣽⣷⣿⣿⣿⣿⢿⣿⣿⣷⣿⣿⣿⣿⣿⣿⣿⣻⣽⣻⢿⣿⣿⣿⣿⣿⣿⢿⣻⢯⣿⣿⣿⣿⣿⣿⡿⣟⣿⣽⣿⣿⣿⣿⣿⣯⣿⣿⣿⣿⣷⣿⣿⣿⣿⣿⡳⠁⠀⠀ ⠀⠀⡰⠀⠀⠀⡎⠀⣾⠁⠀⠀
# ⠀⠀⠀⠸⣇⠀⢣⠀⢀⠈⠆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢻⣿⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣻⣿⣿⣿⣿⢿⣿⣿⣷⣿⣿⣿⡷⣯⢿⣽⣻⣞⣯⡿⣽⣻⢾⣻⡽⣟⣾⣽⣿⣿⣿⣾⣿⣿⣿⣿⣿⣿⡿⣿⣾⣿⣿⣿⣿⣾⣿⣿⢿⣿⡾⡽⠁⠀⠀⠀⠀⠀⠠⠁⠀⠁⡜⠀⣸⠇⠀⠀⠀
# ⠀⠀⠀⠀⢻⡆⠠⢣⠀⢂⠈⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣿⣿⣿⡿⣿⣾⣿⣿⣷⣿⣿⣿⣿⢿⣿⣿⣿⣿⢿⣿⣻⣿⣿⣻⣟⣾⣽⣻⢾⣽⢿⣽⣻⢯⣟⡿⣽⣾⣿⣿⣿⢿⣿⣿⣿⢿⣻⣷⣿⣿⣿⣿⣿⣷⣿⣿⣿⣿⣿⣟⠟⠀⠀⠀  ⠀⠀⡠⠃⠰⠀⡴⠀⢰⡏⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⢻⡄⢂⢢⠀⢂⠈⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢟⣿⣿⣿⣿⣿⣟⣿⣿⣿⣟⣿⣿⣿⣿⡿⣿⣿⣿⣿⣿⣿⣿⣿⣾⣷⣻⣯⣟⡿⣞⣯⣿⣯⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣾⣿⣿⣿⣾⣿⣫⠊⠀⠀⠀⠀ ⠀⠀⠠⠁⠀⠀⡘⠀⢠⡟⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⢻⡆⠀⣧⠀⠀⠈⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣻⣿⣿⣿⣿⣿⣿⢿⣿⣿⣿⣿⣻⣿⣿⣿⣿⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣾⣿⣿⣷⣿⣿⣿⣿⣻⣽⣾⣿⣿⣿⣟⣿⢟⡗⠁⠀⠀⠀⠀   ⠀⠀⠁⠀⠀⡔⠀⣠⡟⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠿⣆⢀⠱⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢾⣻⣿⣿⣽⣾⣿⣿⣿⣟⣿⣿⣿⣿⣟⣿⠟⠉⠀⠀⠈⠉⠉⠉⠉⢹⠉⠉⠉⠉⠉⠁⠀⠀⠉⠻⣿⣿⣿⡿⣿⣿⣷⣿⣿⣿⣿⣿⣿⣾⢿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⢀⠜⠀⣰⠟⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠙⣦⠀⠑⣄⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢯⡻⣿⣿⣿⣿⡿⣿⣿⣿⣿⣽⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣿⣿⣿⣿⣿⣿⣿⢿⣟⣿⣽⢟⡽⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⢠⠊⠀⣴⠏⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢿⣄⠈⢢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⢽⢿⣿⣿⣿⣿⢿⣻⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⣿⣾⣿⣿⡿⣫⠗⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀  ⠀⠀⠀⠀⡔⠁⢠⡾⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣦⠀⠑⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠪⢟⡿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠰⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣷⣿⠿⣫⠗⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀   ⠀⠀⣠⠊⢀⣴⠟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢳⣔⠈⠢⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠒⠫⢽⣻⢿⡀⠀⠀⠀⠀⠀⠀⠀⢀⠠⢺⠐⢄⠀⠀⠀⠀⠀⠀⠀⠀⢠⡿⣟⡫⠝⠊⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀  ⠀⠀⢀⠔⠁⣠⡾⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢷⣄⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⠍⣙⠛⠳⠶⠶⣢⣤⣤⣤⣤⣼⣤⣤⣤⣤⣦⣤⠶⠶⠟⠛⢋⡉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠀⠐⠁⣠⡾⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠀⠒⠒⠂⠠⠤⠤⠤⠤⠤⠤⠄⠀⠒⠒⠈⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀   ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠳⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀     ⠀⠀⠀⠀⠀⠀⢀⣴⠞⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⢶⣄⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠨⢠⠁⢂⡐⠠⠀⠄⠀⢀⠀⡀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⡀⠠⢀⠄⡐⠠⡂⠔⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⢀⠠⠀⠀⣠⡶⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠫⢦⣄⡈⠒⠠⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠂⠡⠍⣈⡔⢂⠢⠐⡁⠆⢂⠔⡠⠌⠂⡅⢊⢁⡂⠅⠆⠐⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠤⠒⢁⣠⡴⠟⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠶⣦⣄⠁⠒⠤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠈⠀⠀⠀⠈⠈⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠒⠈⣀⣤⠶⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠳⢦⣄⣈⠉⠒⠒⠤⠄⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠀⠤⠔⠒⠉⣉⣠⣤⠶⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠛⠳⠶⣤⣤⣀⣀⠉⠉⠉⠐⠒⠒⠒⠀⠒⠠⠄⠠⠄⠒⠂⠒⠒⠒⠂⠈⠉⠉⢀⣀⣤⣤⠶⠞⠛⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠛⠛⠒⠶⠶⠶⠦⣤⣤⣤⣤⣤⣤⣤⣤⠤⠴⠶⠶⠚⠛⠛⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀






class DDPM(nn.Module):
    
    def __init__(self, model: nn.Module, betas: torch.Tensor, 
                numTimesteps: int, dropoutRate: float, device: str, numClasses: int):
        super(DDPM, self).__init__()
        
        self.model = model.to(device)
        self.betas = betas
        self.numTimesteps = numTimesteps
        self.dropoutRate = dropoutRate
        self.numClasses = numClasses
        
        # Define loss in initialization since we don't want to define it for each forward pass
        self.loss = nn.MSELoss()
        self.device = device
        
        # betas are linearly interpolated from the start to the ending beta for each timestep
        # These represent the variance schedule which defines how much noise there is at each timestep
        betas = torch.linspace(betas[0], betas[1], numTimesteps+1)
        sqrtBetas = torch.sqrt(betas)

        # Formula is at the bottom of page 2 in the paper
        # \alpha = 1-\beta_t
        alphas = 1 - betas
        
        ###################################################
        # Training Equations
        ###################################################
        
        # This is the formula found at the bottom of page 2 where \bar{\alpha}_t is the cumulative product of previous \alpha values
        # \bar{alpha}_t = \prod^t_{s=1} \alpha_s
        alphaBar = torch.exp(torch.cumsum(torch.log(alphas), dim=0))
        sqrtComplimentAlphaBar = torch.sqrt(1 - alphaBar)
        
        ###################################################
        # Sampling Equations
        ###################################################
        
        # This is also in equation 11 and is used during sampling
        oneOverSqrtAlphas = 1 / torch.sqrt(alphas)
        
        # This is used in equation 11 of the paper, it represents how much noise is in the current sample at a given timestep
        betasOverSqrtComplimentAlphaBar = betas / sqrtComplimentAlphaBar
        
        self.alphas = alphas.to(device)
        self.oneOverSqrtAlphas = oneOverSqrtAlphas.to(device)
        self.sqrtBetas = sqrtBetas.to(device)
        self.alphaBar = alphaBar.to(device)
        self.sqrtComplimentAlphaBar = sqrtComplimentAlphaBar.to(device)
        self.betasOverSqrtComplimentAlphaBar = betasOverSqrtComplimentAlphaBar.to(device)

    def forward(self, x: torch.Tensor, classLabels: torch.Tensor):
        """
        Sample random timestep for random noise
        """
        
        # Decide what samples should be masked out with 0s or 1s based on the dropout rate
        classMask = torch.bernoulli(torch.full(classLabels.shape, self.dropoutRate)).to(self.device)

        # Select random timesteps in the time range for the forward pass
        randomTimes = torch.randint(1, self.numTimesteps + 1, (x.shape[0],)).to(self.device)
        
        # Random gaussian noise for reparameterization
        noise = torch.randn_like(x)
        
        # This is the forward pass equation from the paper
        scaledSamples = torch.sqrt(self.alphaBar[randomTimes, None, None, None]) * x
        scaledNoise = self.sqrtComplimentAlphaBar[randomTimes, None, None, None] * noise

        noisedSamples = scaledSamples + scaledNoise

        # Loss is MSE between true noise and model predicted noise
        return self.loss(noise, self.model(noisedSamples, classLabels, randomTimes/self.numTimesteps, classMask))
    
    def sample(self, numSamples:int, sampleSize:tuple=(1, 28, 28), classifierGuidance:float=0.5, classLabels:torch.Tensor=None) -> tuple[torch.Tensor, np.ndarray]:
        
        """
        Run sampling from the DDPM by iterating over all timesteps in reverse and denoise the image
        at each timestep.
        
        Arguments:
            numSamples: How many samples to generate
            sampleSize: The shape of the samples to generate
            classifierGuidance: The weight to use for classifier free guidance. Larger means stronger guidance
                which usually leads to less diverse, but more reliable results.
            classLabels: An optional 1D tensor of labels to generate from
            
        Returns:
            (generatedSamples, intermediateSamples)
            generatedSamples: A (N, 1, 28, 28) tensor of generated samples
            intermediateSamples: An array of intermediate generated samples
        """
        
        # Condition on every class evenly unless provided with class labels to sample from
        if not classLabels:
            classLabels = torch.arange(0, self.numClasses).to(self.device)
        
        # Define the shape of noisy samples to start the denoising process
        noiseShape = (numSamples, *sampleSize)
        
        # Start out our samples as pure noise
        noisySamples = torch.randn(*noiseShape).to(self.device)
        classLabels = classLabels.repeat(int(numSamples / classLabels.shape[0])).repeat(2)

        # Create a class mask of 0s in the first half and 1s in the second half.
        # We will use this for CFG where we want the model to learn without the class label
        classMask = torch.zeros(classLabels.shape).to(self.device)
        classMask[numSamples:] = 1
        
        # Store intermediate samples for plotting
        sampleStorage = []
        for time in range(self.numTimesteps, 0, -1):
            currentTimestep = torch.tensor([time / self.numTimesteps]).repeat(numSamples, 1, 1, 1).to(self.device)
            
            # Expand dimensions of samples and timesteps so we can do CFG later
            noisySamples = noisySamples.repeat(2,1,1,1)
            currentTimestep = currentTimestep.repeat(2,1,1,1)
            
            # Get model predictions for noise when given class labels and times
            predictedNoise = self.model(noisySamples, classLabels, currentTimestep, classMask)
            
            # Use classifier free guidance by shifting predictions 
            # in the directions of guided samples
            unguidedNoisePredictions = predictedNoise[numSamples:]
            guidedNoisePredictions = predictedNoise[:numSamples]
            
            # We say the real predicted noise is noise pushed in the direction of the guided samples as opposed to the
            # unguided predictions. This helps improve the performance of the diffusion process at the cost of potentially
            # generating worse samples if the guidance isn't tuned properly.
            predictedNoise = (1 + classifierGuidance) * guidedNoisePredictions - classifierGuidance * unguidedNoisePredictions
            
            # Create random noise for sample denoising
            if time > 1:
                z = torch.randn(*noiseShape).to(self.device)
            else:
                z = 0
            
            # Pick out the guided noisy samples and denoise them using predicted 
            # noise and the appropriate scaling from the DDPM beta scheduling
            denoisedSamples = noisySamples[:numSamples] - predictedNoise * self.betasOverSqrtComplimentAlphaBar[time]
            noisySamples = self.oneOverSqrtAlphas[time] * denoisedSamples + self.sqrtBetas[time] * z
            
            # TODO: Normalize stored samples? They appear very faded when looking at results
            if (time < 10) or (time % 20 == 0) or (time == self.numTimesteps):
                sampleStorage.append(noisySamples.detach().cpu().numpy())
            
        # sampleStorage = np.array(sampleStorage)
        # return noisySamples, sampleStorage
        return noisySamples, torch.Tensor(np.array(sampleStorage)) # Convert from list to array to tensor because pytorch says that's faster somehow



def writeIntermediateResults(ddpm:DDPM, guidanceStrengths:list, savePath:str, label:str, ep:int, nc:int, ts:int):

    def createGif(imageList, outPath, duration=100):
        imageList[0].save(outPath, save_all=True, append_images=frames[1:], duration=duration, loop=0)


    numClasses = ddpm.numClasses

    ddpm.eval()
    with torch.no_grad():
        for w in guidanceStrengths:
            
            dataPath = savePath + f'{label}_ep-{ep}_w-{w}_nc-{nc}_ts-{ts}'
            
            imFilename = dataPath + '.png'
            gifFilename = dataPath + '.gif'

            # intermediateResults is shape (F, B, 1, 28, 28) where F is frame count
            generated, intermediateResults = ddpm.sample(numClasses*8, (1, 28, 28), classifierGuidance=w)

            # TODO: Fix this, we want ncol=numClasses but there isn't a ncol kwarg
            imGrid = make_grid(generated, nrow=numClasses, normalize=True)
            save_image(imGrid, fp=imFilename)
            
            frames = []
            for frame in intermediateResults:
                
                imGrid = make_grid(frame, nrow=numClasses, normalize=True)
                img = transforms.ToPILImage()(imGrid)
                frames.append(img)
                
            createGif(frames, gifFilename, 100)





def trainDDPM(model, numClasses: int, epochs: int, batch_size: int, numTimesteps: int, dataset: Dataset, label: str, transform=transforms.Compose([]), betas=(1e-4, 0.02)):
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Enable or disable automatic mixed precision for faster training
    USE_AMP = True
    lr = 1e-4
    saveModel = True
    
    label = f'{label}_ep-{epochs}_BS-{batch_size}_ts-{numTimesteps}_bt-{betas}'
    
    savePath = f'./DiffusionData/{label}/'
    
    if not os.path.isdir(savePath):
        os.mkdir(savePath)
    
    # CFG Guidance strengths
    guidanceStrengths = [0, 1, 2]
    
    ddpm = DDPM(model=model, betas=betas, numTimesteps=numTimesteps, dropoutRate=0.4, device=device, numClasses=numClasses)
    ddpm.to(device)

    # TODO: Try different optimizers
    # NOTE: DO NOT CHANGE num_workers>0 OR THE MODEL WON'T FUCKING TRAIN!!!!. WHY THE FUCK DOES THIS HAPPEN???!??!?!??!?!??!?!??!??!?!!!??!??!?!?
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)
    lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, factor=0.75, patience=40, cooldown=40)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)


    pbar = tqdm(range(epochs))
    for ep in pbar:
        
        ddpm.train()
        newLR = optim.param_groups[0]['lr']

        for features, labels in dataloader:
            
            # Apply transform manually on the GPU
            features = transform(features)
            labels = labels

            optim.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                loss = ddpm(features, labels)
            
            scaler.scale(loss).backward() # Do backpropagation on scaled loss from AMP
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=5)
            scaler.step(optim)
            scaler.update()
            
            pbar.set_description(f"loss: {loss.item():.4f}, lr: {newLR:.6f}")

        lrScheduler.step(loss.item())

        if (ep % 200 == 0 or ep == int(epochs-1)) and (ep != 0):
            writeIntermediateResults(ddpm, guidanceStrengths, savePath=savePath, label=label, ep=ep, nc=numClasses, ts=numTimesteps)

        # Save model
        if saveModel and ep == epochs-1:
            modelName = f"model-{label}.pth"
            torch.save(ddpm.state_dict(), savePath + modelName)


def fineTuneDDPM(ddpm, numClasses: int, epochs: int, batch_size: int, numTimesteps: int, dataset: Dataset, label: str, transform=transforms.Compose([]), betas=(1e-4, 0.02)):
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Enable or disable automatic mixed precision for faster training
    USE_AMP = True
    lr = 1e-4
    saveModel = True
    
    label = f'{label}_ep-{epochs}_BS-{batch_size}_ts-{numTimesteps}_bt-{betas}'
    
    savePath = f'./DiffusionData/{label}/'
    
    if not os.path.isdir(savePath):
        os.mkdir(savePath)
    
    # CFG Guidance strength
    guidanceStrengths = [0, 1, 2]
    ddpm.to(device)

    # TODO: Try different optimizers
    # NOTE: DO NOT CHANGE num_workers>0 OR THE MODEL WON'T FUCKING TRAIN!!!!. WHY THE FUCK DOES THIS HAPPEN???!??!?!??!?!??!?!??!??!?!!!??!??!?!?
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)
    lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, factor=0.75, patience=40, cooldown=40)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)


    pbar = tqdm(range(epochs))
    for ep in pbar:
        
        ddpm.train()
        newLR = optim.param_groups[0]['lr']

        for features, labels in dataloader:
            
            # Apply transform manually on the GPU
            features = transform(features)
            labels = labels

            optim.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                loss = ddpm(features, labels)
            
            scaler.scale(loss).backward() # Do backpropagation on scaled loss from AMP
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), max_norm=5)
            scaler.step(optim)
            scaler.update()
            
            pbar.set_description(f"loss: {loss.item():.4f}, lr: {newLR:.6f}")

        lrScheduler.step(loss.item())

        if (ep % 200 == 0 or ep == int(epochs-1)) and (ep != 0):
            writeIntermediateResults(ddpm, guidanceStrengths, savePath=savePath, label=label, ep=ep, nc=numClasses, ts=numTimesteps)

        # Save model
        if saveModel and ep == epochs-1:
            modelName = f"model-{label}.pth"
            torch.save(ddpm.state_dict(), savePath + modelName)








def main():
    
    from torchvision.datasets import MNIST
    
    trainDDPM(numClasses=10, epochs=2, batch_size=128, numTimesteps=400, dataset=MNIST('./Data', transform=transforms.ToTensor()))
    
    
if __name__ == '__main__':
    main()


