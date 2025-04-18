# -*- coding: utf-8 -*-
"""Exercise 2: Uncertainty Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14ndkHlyFq3ObjzLSyYcE9HMkoNdqAWPA
"""

# mount your google drive to load files directly from there
from google.colab import drive
drive.mount('/content/drive')

# if you want to import Python files, add the correct Google Drive directory to your Pythonpath
import sys
sys.path.append('/content/drive/My Drive/work/teaching/2025 Advanced Deep Learning/')

# Download the data from huggingface (https://huggingface.co/datasets/simbaswe/galah4/tree/main)
# and upload it to your google drive. Then, specify this directory here
import os

DATA_PATH = "/content/drive/My Drive/Colab Notebooks/Data Astronomy CNN with PyTorch" # MY SAVED PATH
print(os.listdir(DATA_PATH)) # print to see that it found the files

# import the stuff you need. Pytorch is already installed on Google colab
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from matplotlib import pyplot as plt
from torchsummary import summary

spectra = np.load(f"{DATA_PATH}/spectra.npy")
spectra_length = spectra.shape[1]
# labels: mass, age, l_bol, dist, t_eff, log_g, fe_h, SNR
labelNames = ["mass", "age", "l_bol", "dist", "t_eff", "log_g", "fe_h", "SNR"]
labels = np.load(f"{DATA_PATH}/labels.npy")

# We only use the three labels: t_eff, log_g, fe_h
labelNames = labelNames[-4:-1]
labels = labels[:, -4:-1]
n_labels = labels.shape[1]

print(labels) # inspect data labels

from sklearn.preprocessing import MinMaxScaler

# normalize the spectra and labels via log
spectra = np.log(np.maximum(spectra, 0.2))

# scale all labels with minmaxscaler independently and keep the parameters for "unscaling"
scaler = MinMaxScaler()
labels = scaler.fit_transform(labels)

print(labels) # inspect the labels indeed transformed correctly

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# plot a few spectra
num_spectra = 1
for i in range(num_spectra):
  fig, ax = plt.subplots(1, 1, figsize=(10, 5))
  ax.plot(spectra[i], lw=1)
  ax.set_title(f"Star {i}")
  ax.set_xlabel("Wavelength")
  ax.set_ylabel("Flux")
  plt.tight_layout()

"""#Create training data (the efficient way)"""

from torch.utils.data import Dataset, DataLoader, random_split

class SpectraDataset(Dataset):
    def __init__(self, spectra, labels):
        self.spectra = spectra # input: shape [--,--,--]
        self.labels = labels # output: shape [--,--]

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        return self.spectra[idx], self.labels[idx] # returns corresponding datapoint

# Convert to PyTorch tensors
spectra_tensor = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)  # Add a channel dimension for the CNN later on
labels_tensor = torch.tensor(labels, dtype=torch.float32)

dataset = SpectraDataset(spectra_tensor, labels_tensor) # pass to "datamaker"


split_ratio = 0.1

total = len(dataset)
val_size = int(0.1 * total)
train_size = int(0.8 * total)
test_size = total - train_size - val_size  # Guarantee all samples are used

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders --> indices (see lecture notes). Dataloader makes it easy to work with the data in terms of loading from memory as needed
# and convinient to iterate over and pass to our model
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Total dataset size:", len(dataset))
print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Test size:", len(test_dataset))  # This should be ~670

"""#Implement CNN in pytorch
i.e. we need to first create a class for the model. we require 3 outputs: T_eff, log_g, Fe/H.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 3 μ + 3 log σ outputs
        )

    def forward(self, x):  # x shape: [B, 1, 16384]
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        out = self.fc(x)
        mu = out[:, :3]
        log_sigma = out[:, 3:]
        return mu, log_sigma  # or also return torch.exp(log_sigma) if you prefer. I don't.

# === Custom NLL Loss Function ===
def nll_loss(mu, log_sigma, y):
    """we use this loss since we derived it from the max likelihood principle. It incooperates
    the values for the uncertainties as well, given that they behave as gaussians, whihch we assume
    in this exercise."""

    std = torch.exp(log_sigma)                  # Convert log std to std

    # NLL formula for Gaussian likelihood
    return torch.mean(0.5 * ((y - mu) / std) ** 2 + log_sigma)

"""# Setup Tensorboard for monitoring and train the model"""

# Commented out IPython magic to ensure Python compatibility.
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm  # for progress bar
from torch.utils.tensorboard import SummaryWriter


# === Setup device and tensorboard ===
writer = SummaryWriter("runs/UncertaintyNet")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %load_ext tensorboard


# === Setup model, optimizer, scheduler ===
model = UncertaintyNet().to(device)  # or UncertaintyNet() if using MLP
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)


# === Training config ===
num_epochs = 30
patience = 3
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []

# === Training loop ===
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device) # See comment on this line below [1]

        optimizer.zero_grad()

        mu, log_sigma = model(x_batch)  # unpack
        loss = nll_loss(mu, log_sigma, y_batch)

        writer.add_scalar("nll", loss.item(), epoch) # log to tensorboard using the obtained loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x_batch.size(0)

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # === Log weights and biases ===
    for name, param in model.named_parameters():
      writer.add_histogram(name, param, epoch)


    # === Validation loop ===
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)

            mu, log_sigma = model(x_val)
            loss = nll_loss(mu, log_sigma, y_val)

            val_loss += loss.item() * x_val.size(0)

    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    # scheduler.step(val_loss)  # optional but useful

    # === Early stopping ===
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")


writer.close() # close tensorboard logging
# %tensorboard --logdir runs

# -----------------------------------------------------------------------------------------

# [1] this is indeed not optimal to move each batch to the device. Since we are working with a relatively small dataset we can get away
# with this. However, I am not completely sure how to move the entire dataset directly. Co-pilot instructed me to implement


# train_dataset.data = train_dataset.data.to(device)
#     train_dataset.targets = train_dataset.targets.to(device)
#     val_dataset.data = val_dataset.data.to(device)
#     val_dataset.targets = val_dataset.targets.to(device)
#     test_dataset.data = test_dataset.data.to(device)
#     test_dataset.targets = test_dataset.targets.to(device)

# But I am not sure this is the best way of doing it. If you (the reader) have a better way I am curious to know.

# === Testing ===
model.eval()
examples = 3
test_loss = 0.0

all_mu = []
all_sigma = []
all_targets = []

with torch.no_grad():
    for batch_idx, (x_test, y_test) in enumerate(test_loader):
        x_test, y_test = x_test.to(device), y_test.to(device)

        mu, log_sigma = model(x_test)
        sigma = torch.exp(log_sigma)

        loss = nll_loss(mu, log_sigma, y_test)
        test_loss += loss.item() * x_test.size(0)

        # Always append full batch
        all_mu.append(mu.detach().cpu())
        all_sigma.append(sigma.detach().cpu())
        all_targets.append(y_test.detach().cpu())

        # print a few from first batch only
        if batch_idx == 0:
            mu_np = mu.detach().cpu().numpy()
            sigma_np = sigma.detach().cpu().numpy()
            y_np = y_test.detach().cpu().numpy()

            for i in range(min(examples, len(mu_np))):
                print(f"Example {i+1}:")
                print(f"Predicted μ: {mu_np[i]}  Predicted σ: {sigma_np[i]}")
                print(f"Actual:      {y_np[i]}")
                print("-" * 50)

test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss}")

"""# Unscale predictions"""

# Stack all batches
mu_all = torch.cat(all_mu).numpy()           # shape [N, 3]
sigma_all = torch.cat(all_sigma).numpy()     # shape [N, 3]
y_all = torch.cat(all_targets).numpy()       # shape [N, 3]

# Inverse transform
mu_unscaled = scaler.inverse_transform(mu_all)
actuals_unscaled = scaler.inverse_transform(y_all)

# Rescale sigma
label_range = scaler.data_max_ - scaler.data_min_
sigma_unscaled = sigma_all * label_range

"""#Plot resulting loss (not really necassary since we can see it in tensorboard, but i did this before implementing tensorboard. However, it's good practice)"""

# After training
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.legend()
plt.grid(True)
plt.show()

"""#Scatterplot inspections and residual histograms"""

labels = ['Teff', 'log(g)', '[Fe/H]']

plt.figure(figsize=(18, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)

    # Scatter plot
    plt.scatter(actuals_unscaled[:, i], mu_unscaled[:, i], alpha=0.5, label='Predictions')

    # Red y = x line
    min_val = min(actuals_unscaled[:, i].min(), mu_unscaled[:, i].min())
    max_val = max(actuals_unscaled[:, i].max(), mu_unscaled[:, i].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')

    # Labels and formatting
    plt.xlabel(f'True {labels[i]}')
    plt.ylabel(f'Predicted {labels[i]}')
    plt.title(f'{labels[i]}: Prediction vs. True')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Compute residuals
residuals = mu_unscaled - actuals_unscaled  # shape: [N, 3]
labels = ['Teff', 'log(g)', '[Fe/H]']

# Plot histograms
plt.figure(figsize=(18, 5))
for i in range(3):
    mean_res = np.mean(residuals[:, i])
    std_res = np.std(residuals[:, i])

    plt.subplot(1, 3, i + 1)
    plt.hist(residuals[:, i], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', label='Zero Error')
    plt.axvline(mean_res, color='green', linestyle='--', label='Mean')
    plt.title(f"Residuals for {labels[i]}\nMean ≈ {mean_res:.2f}, Std ≈ {std_res:.2f}")
    plt.xlabel(f"Predicted − True {labels[i]}")
    plt.ylabel("Count")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

"""#Plot Pull Distrubutions"""

# Stack all batches
mu_all = torch.cat(all_mu).numpy()          # shape: [N, 3]
sigma_all = torch.cat(all_sigma).numpy()    # shape: [N, 3]
y_all = torch.cat(all_targets).numpy()      # shape: [N, 3]

# Compute pull
pull = (mu_all - y_all) / sigma_all         # shape: [N, 3]

# Plot pull distribution for each label
labels = ["$T_{eff}$", "$\log g$", "$Fe/H$"]
plt.figure(figsize=(18, 5))

for i in range(3):
    mean_pull = np.mean(pull[:, i])
    std_pull = np.std(pull[:, i])

    plt.subplot(1, 3, i + 1)
    plt.hist(pull[:, i], bins='auto', alpha=0.75, color='skyblue', edgecolor='black', density=True)
    plt.axvline(0, color='black', linestyle='--', label='Zero Pull')
    plt.axvline(mean_pull, color='green', linestyle='--', label='Mean Pull')
    plt.title(f"Pull Distribution: {labels[i]}\nMean ≈ {mean_pull:.2f}, Std ≈ {std_pull:.2f}")
    plt.xlabel("Pull")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()

"""#Final thoughts

### Training
We see that the training, early stopping is triggered after 26 epochs. At this point we have obtained good result which we see at the prints. The prints looks relatively close to the true value, indicating the model works appropriately.

### Pull Distrubutions
We see that the standard deviations all follow a gaussian with a mean zero (or close to). Since they are not particularly wide or thin, we conclude that the the prediction are resonable. We can summarize this as: If the pull distribution:

* is wider than standard normal (e.g., std > 1) → the uncertainties are too small (underconfident)

* is narrower (e.g., std < 1) → the uncertainties are too large (overconfident)

* is shifted from 0 → the model is biased. This can be argued to be the case for $T_{eff}$, however, this is a relatively small effect and the std looks really good! So overall the results are good I would argue.

### Scaling
Notice that we have trained and tested the model on the scaled data. This process can always be inverted via

\begin{equation}
  y_{\text{original}} = y_{\text{scaled}} \cdot (y_{\text{max}} - y_{\text{min}}) + y_{\text{min}},
\end{equation}

for minmaxscaler for *labels* and for the *spectra* via spectra_inverted = np.exp(spectra_transformed). This I have easily done by labels_unscaled = scaler.inverse_transform(labels_scaled) as implemented.


### Negative Loss
We notice that we have negative loss during training and testing. This we are not used to when using *mse*. However, this is not a problem since we have used a modified loss-function Negative log-likelihood (*NLL*). We can always add a constant to this to keep it postive, however, the negativity is not an inherent problem. This we can back up by the following observations:

* The predictions make sense (plot a few μ ± σ intervals)

* The validation loss continues to track training loss

* There's no overfitting or instability

Then this is fine. The negative loss just means the model is improving in *log-probability space*.





"""