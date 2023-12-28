import sqlite3
import faiss
import torch
import numpy as np
from natsort import natsorted
import os
from torchvision import models
from ImageDataset import NpyDataset
from SimsiamSameEncoderModel import SimSiam
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
from Meta_Data_Extracter import get_planet_meta_data, get_maxar_meta_data
from shapely.geometry import Polygon
from torch.utils.data import DataLoader, random_split
from Transforms import get_transform
from PlotConfusionMatrix import plot_mertics
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from PlotConfusionMatrix import plot_mertics

def dummy_plotters():
    actuals = []
    preds = []
    for i in range(50):
        actuals.append(0)
        preds.append(0)
    
    for i in range(50):
        actuals.append(0)
        preds.append(2)
    
    for i in range(100):
        actuals.append(1)
        preds.append(1)

    for i in range(70):
        actuals.append(2)
        preds.append(2)
    
    for i in range(30):
        actuals.append(2)
        preds.append(0)
    
    plot_mertics(actuals, preds, "Two Encoder Based")
    precision = precision_score(actuals, preds, average="macro")
    recall = recall_score(actuals, preds, average="macro")
    f1 = f1_score(actuals, preds, average="macro")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

dummy_plotters()