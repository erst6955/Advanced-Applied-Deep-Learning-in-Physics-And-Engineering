import time
import sys
import os
import argparse
import io
from datetime import datetime
import numpy as np
import awkward


# ======== Import if if we want to split up our files=======
from gnn_encoder import GNNEncoder, collate_fn_gnn
from gnn_trafo_helper import train_model, evaluate_model, normalize, denormalize, get_img_from_matplotlib, get_information, plot_predictions, plot_loss_curve, normalize_dataset, plot_residuals

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph, DynamicEdgeConv, global_mean_pool



 # ================================================ Initialization And Normalization =========================================================
DATA_PATH = r"C:\Users\eriks\OneDrive\Desktop\Exercise 4 - Graph Neural Networks\Dataset"  # path to the data

# Load the dataset
train_dataset = awkward.from_parquet(os.path.join(DATA_PATH, "train.pq"))
val_dataset = awkward.from_parquet(os.path.join(DATA_PATH, "val.pq"))
test_dataset = awkward.from_parquet(os.path.join(DATA_PATH, "test.pq"))

get_information(train_dataset, val_dataset, test_dataset)

# Calculate means and standard deviations for normalization
time_mean = np.mean(train_dataset["data"][:, 0, :])
time_std = np.std(train_dataset["data"][:, 0, :])
x_mean = np.mean(train_dataset["data"][:, 1, :])
x_std = np.std(train_dataset["data"][:, 1, :])
y_mean = np.mean(train_dataset["data"][:, 2, :])
y_std = np.std(train_dataset["data"][:, 2, :])

# Normalize all datasets using the same stats (computed from training set)
train_dataset = normalize_dataset(train_dataset, time_mean, time_std, x_mean, x_std, y_mean, y_std)
val_dataset   = normalize_dataset(val_dataset, time_mean, time_std, x_mean, x_std, y_mean, y_std)
test_dataset  = normalize_dataset(test_dataset, time_mean, time_std, x_mean, x_std, y_mean, y_std)



 # =================================================================== Run Code ========================================================================    
batch_size = 32
epochs = 2


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_gnn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_gnn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_gnn)

print(f"Train data set size: {len(train_dataset)} | Test data set size: {len(test_dataset)} | Validation data set size: {len(val_dataset)} | Batch size: {batch_size}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
input_dim = 3  # time, x, y
hidden_layers = [64, 64, 64]  # hidden layer dimensions
k = 30  # number of nearest neighbors
output_dim = 2  # predicting xpos and ypos


# ================================== Model Definition And Training =========================================================
model = GNNEncoder(hidden_layers, k, input_dim, output_dim).to(device)

criterion = nn.MSELoss()

model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, device, epochs)
test_loss = evaluate_model(model, test_loader, criterion, device)


# ================================= Plotting And Evaluation =========================================================
plot_loss_curve(train_losses, val_losses, test_loss)
plot_predictions(model, test_loader, x_mean, x_std, y_mean, y_std, device)
plot_residuals(model, test_loader, x_mean, x_std, y_mean, y_std, device)






"""
lika stora?
"""







