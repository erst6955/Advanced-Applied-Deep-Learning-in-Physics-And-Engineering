


import torch
import numpy as np
from matplotlib import pyplot as plt
import io
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import time
import awkward
from scipy.stats import norm


# ================================================== Training And Normalization =========================================================
def normalize_dataset(dataset, time_mean, time_std, x_mean, x_std, y_mean, y_std):
    """
    Normalizes the data and labels of an Awkward dataset in-place.
    Normalize data and labels
    working with Awkward arrays is a bit tricky because the ['data'] field can't be assigned in-place,
    so we need to extract the time, x, and y coordinates, normalize them separately,
    and then concatenate them back together.
    """
    times = dataset["data"][:, 0:1, :] # important to index the time dimension with 0:1 to keep this dimension (n_events, 1, n_hits)
                                        # with [:,0,:] we would get a 2D array of shape (n_events, n_hits)
    norm_times = normalize(times, time_mean, time_std)

    x = dataset["data"][:, 1:2, :]
    norm_x = normalize(x, x_mean, x_std)

    y = dataset["data"][:, 2:3, :]
    norm_y = normalize(y, y_mean, y_std)

    # Combine normalized features
    dataset["data"] = awkward.concatenate([norm_times, norm_x, norm_y], axis=1)

    # Normalize labels
    dataset["xpos"] = normalize(dataset["xpos"], x_mean, x_std)
    dataset["ypos"] = normalize(dataset["ypos"], y_mean, y_std)

    return dataset



def train_model(model, train_loader, val_loader, criterion, device, num_epochs):

    model.to(device)

    min_delta=0.05
    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    early_stopping_patience = 10
    reduce_lr_patience = 3
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=reduce_lr_patience, verbose=True)


    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        total_train_loss = 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Evaluate on validation set
        val_loss = evaluate_model(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - start_time


        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                break
        

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | lr: {lr} | ES: {epochs_no_improve} | Time: {elapsed:.2f}s")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def normalize(x, mean, std):
    return (x - mean) / std

def denormalize(x, mean, std):
    return x * std + mean

def get_img_from_matplotlib(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    return buf

def get_information(train_dataset, val_dataset, test_dataset):
    # to get familiar with the dataset, let's inspect it.
    print(f"The training dataset contains {len(train_dataset)} events.")
    print(f"The validation dataset contains {len(val_dataset)} events.")
    print(f"The test dataset contains {len(test_dataset)} events.")
    print(f"The training dataset has the following columns: {train_dataset.fields}")
    print(f"The validation dataset has the following columns: {val_dataset.fields}")
    print(f"The test dataset has the following columns: {test_dataset.fields}")
    # print the first event of the training dataset
    print(f"The first event of the training dataset is: {train_dataset[0]}")

    # We are interested in the labels xpos and ypos. This is the position of the neutrino interaction that we want to predict.
    print(f"The first event of the training dataset has the following labels: {train_dataset['xpos'][0]}, {train_dataset['ypos'][0]}")
    # Awkward arrays also allow us to obtain the 'xpos' and 'ypos' label for all events in the dataset
    print(f"The first 10 labels of the training dataset are: {train_dataset['xpos'][:10]}, {train_dataset['ypos'][:10]}")

    # The data can be accessed by using the 'data' key.
    # The data is a 3D array with the first dimension being the number of events,
    # the second dimension being the the three features (time, x, y)
    # the third dimension being the number of hits,
    print(f"The first event of the training dataset has {len(train_dataset['data'][0][0])} hits, i.e., detected photons.")
    # Let's loop over all hits and print the time, x, and y coordinates of the first event.
    for i in range(len(train_dataset['data'][0, 0])):
        print(f"Hit {i}: time = {train_dataset['data'][0,0,i]}, x = {train_dataset['data'][0,1, i]}, y = {train_dataset['data'][0,2,i]}")
    # To get all hit times of the first event, you can use the following code:
    print(f"The first event of the training dataset has the following hit times: {train_dataset['data'][0, 0]}")
    print(f"The first event of the training dataset has the following hit x positions: {train_dataset['data'][0, 1]}")
    print(f"The first event of the training dataset has the following hit y positions: {train_dataset['data'][0, 2]}")




 # =================================================================== Plotting ========================================================================
def plot_predictions(model, test_loader, x_mean, x_std, y_mean, y_std, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            preds.append(output.cpu())
            targets.append(labels.cpu())

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    # Denormalize predictions and targets
    preds_x = denormalize(preds[:, 0], x_mean, x_std)
    preds_y = denormalize(preds[:, 1], y_mean, y_std)

    true_x = denormalize(targets[:, 0], x_mean, x_std)
    true_y = denormalize(targets[:, 1], y_mean, y_std)  

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(true_x, true_y, alpha=0.5, label="True Position")
    plt.scatter(preds_x, preds_y, alpha=0.5, label="Predicted Position")
    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")
    plt.title("True vs. Predicted Neutrino Positions")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.savefig("predictions.png", dpi=300)
    plt.show()


def plot_loss_curve(train_losses, val_losses, test_loss=None):
    """
    Plots training and validation loss over epochs.
    Optionally adds test loss as a horizontal dashed red line.

    Args:
        train_losses (list): Training loss per epoch
        val_losses (list): Validation loss per epoch
        test_loss (float, optional): Final test loss to show as a horizontal line
    """
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    
    if test_loss is not None:
        plt.axhline(test_loss, color="red", linestyle="--", label="Test Loss")

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss.png", dpi=300)
    plt.show()


def plot_residuals(model, data_loader, x_mean, x_std, y_mean, y_std, device):
    """
    Plots residuals (prediction error) for x and y positions with Gaussian fits.
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            preds.append(output.cpu())
            targets.append(labels.cpu())

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    # Denormalize
    preds_x = denormalize(preds[:, 0], x_mean, x_std)
    preds_y = denormalize(preds[:, 1], y_mean, y_std)
    true_x = denormalize(targets[:, 0], x_mean, x_std)
    true_y = denormalize(targets[:, 1], y_mean, y_std)

    # Residuals
    residuals_x = (preds_x - true_x).numpy()
    residuals_y = (preds_y - true_y).numpy()

    # Fit Gaussians
    mu_x, std_x = norm.fit(residuals_x)
    mu_y, std_y = norm.fit(residuals_y)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for res, mu, std, ax, label, color in zip(
        [residuals_x, residuals_y],
        [mu_x, mu_y],
        [std_x, std_y],
        axes,
        ["x", "y"],
        ["royalblue", "orange"]
    ):
        n, bins, _ = ax.hist(res, bins='auto', density=True, alpha=0.6, color=color, label=f"{label} residual")
        pdf = norm.pdf(bins, mu, std)
        ax.plot(bins, pdf, "k--", label=f"Gaussian fit\nμ={mu:.2f}, σ={std:.2f}")
        ax.set_title(f"Residuals in {label}")
        ax.set_xlabel(f"{label} residual [m]")
        ax.set_ylabel("Density")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig("residual.png", dpi=300)
    plt.show()

    print(f"Residuals x: mean = {mu_x:.3f}, std = {std_x:.3f}")
    print(f"Residuals y: mean = {mu_y:.3f}, std = {std_y:.3f}")
