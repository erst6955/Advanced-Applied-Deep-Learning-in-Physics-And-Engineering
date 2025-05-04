import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence # http://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
import os
import numpy as np
import awkward
from torch.utils.data import DataLoader
from gnn_trafo_helper import normalize_dataset, normalize, denormalize, plot_predictions, plot_loss_curve, plot_pull, train_model, evaluate_model



# ==== LOAD & NORMALIZE DATA ====
DATA_PATH = r"C:\Users\Erik\Desktop\Advanced Applied Deep Learning In Physics And Engineering\Exercises\Exercise 8 - Transformers for IceCube2D\Dataset"
print(os.listdir(DATA_PATH))

train_dataset = awkward.from_parquet(os.path.join(DATA_PATH, "train.pq"))
val_dataset   = awkward.from_parquet(os.path.join(DATA_PATH, "val.pq"))
test_dataset  = awkward.from_parquet(os.path.join(DATA_PATH, "test.pq"))

# Compute normalization stats on training data
time_mean = np.mean(train_dataset["data"][:, 0, :])
time_std  = np.std(train_dataset["data"][:, 0, :])
x_mean    = np.mean(train_dataset["data"][:, 1, :])
x_std     = np.std(train_dataset["data"][:, 1, :])
y_mean    = np.mean(train_dataset["data"][:, 2, :])
y_std     = np.std(train_dataset["data"][:, 2, :])

# Normalize all datasets
train_dataset = normalize_dataset(train_dataset, time_mean, time_std, x_mean, x_std, y_mean, y_std)
val_dataset   = normalize_dataset(val_dataset, time_mean, time_std, x_mean, x_std, y_mean, y_std)
test_dataset  = normalize_dataset(test_dataset, time_mean, time_std, x_mean, x_std, y_mean, y_std)


def collate_fn_transformer(batch):
    """
    Custom function that defines how batches are formed.

    To process the batch items that each have a different number of hits, it is efficient
    to first concatenate all the data into a single tensor and save the lengths of each
    individual event to be able to split the data again later.

    # F: input_dim, number of features (time, x, y)
    # N: number of hits (different for each event)
    # B: batch size

    The resulting 2D tensor has the shape (B x N, F) where B is the batch size, N is the total number of hits of all events
    in the batch, and F is the number of features (time, x, y).


    Parameters
    ----------
    batch : list
        A list of dictionaries containing the data and labels for each graph.
        The data is available in the "data" key and the labels are in the "xpos" and "ypos" keys.
    Returns
    -------
    packed_data : Batch
        A batch of graph data objects.
    labels : torch.Tensor
        A tensor containing the labels for each graph.
    """
    data_list = []
    labels = []
    lengths=[]

    for b in batch:
        # this is a loop over each event within the batch
        # b["data"] is the first entry in the batch with dimensions (n_features, n_hits)
        # where the feautures are (time, x, y)
        tensordata = torch.from_numpy(b["data"].to_numpy()).T
        # the original data is in double precision (float64), for our case single precision is sufficient
        # we let's convert to single precision (float32) to save memory and computation time
        tensordata = tensordata.to(dtype=torch.float32)

        lengths.append(tensordata.shape[0])

        data_list.append(tensordata)

        # also the labels need to be packaged as pytorch tensors
        labels.append(torch.Tensor([b["xpos"], b["ypos"]]).unsqueeze(0))

    labels = torch.cat(labels, dim=0) # convert the list of tensors to a single tensor

    data_vec=torch.cat(data_list) # (B, N, F)  -> (BxN, F) where B is the batch size, N is the number of hits, and F is the number of features (time, x, y)

    # return a list [datalist, lengths]
    return [data_vec, lengths], labels


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=3, d_model=128, nhead=2, num_layers=2, dim_feedforward=256, output_dim=2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model) # input embedding in to dim (B x N, D)

        encoder_layer = nn.TransformerEncoderLayer( # main transformer part
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="relu",
            batch_first=True,
            norm_first=True,
            dropout=0.02
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) 
        self.output_layer = nn.Linear(d_model, output_dim) # Output layer to predict (x, y)


    def forward(self, data) -> torch.Tensor:
        """
        Args:
            data: list of (src tensor, lengths)
        Returns:
            Tensor of shape (batch, output_dim)
        """

        src, lengths = data 

        # F: input_dim, number of features (time, x, y)
        # N: number of hits
        # D: hidden_dim, internal transformer computing dimension
        # B: batch size

        # 1) embed the input data into the hidden dimension
          # shape (B x N, F) -> (B x N, D)
        src = self.input_proj(src)

        # 2) split the data into a list of tensors, one for each event
        parts = src.split(lengths, dim=0)  # shape (B x N, D) -> (B, N, D), where every batch entry can have a variable length,
                                           # i.e., list of tensors of shape (N_i, D) where N_i is the number of hits in the i-th event


        # 3) pad inputs with zeros so that all batch items have same length
        # the Transformer doesn’t automatically know which values are real and which are padding!
        padded = pad_sequence(parts, batch_first=True) # shape (B, N, D) -> (B x MAXLEN x D) now all batch entries have the same length
        batch_size, max_len, _ = padded.shape

        # 4) build the padding mask (batch_size, max_len)
        # we need to keep track which tokens are padding tokens and which are real tokens
        # the mask is a boolean tensor of shape (B, MAXLEN) where True indicates that the corresponding entry is a padding token
        # and False indicates that the corresponding entry is a real token
        # the mask is used to ignore the padding tokens in the attention mechanism
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool).to(device=padded.device, dtype=torch.bool)
        for i, L in enumerate(lengths):
            mask[i, L:] = True  # sets the remaining entries in the row (after the real data) to True

        # 5) call the transformer with padded tensor of shape (B, MAXLEN, D) and corresponding mask of shape (B, MAXLEN)
        enc_out = self.encoder(padded, src_key_padding_mask=mask) # output: (batch_size, max_len, d_model) 

        # 6) masked mean‐pool, i.e., form the average for every batch item along the sequence dimension
        # the output of the transformer is a tensor of shape (B, MAXLEN, D)
        # we need to take the mean over the sequence dimension (MAXLEN) to get a single vector for each batch item
        # we need to ignore the padding tokens in the mean pooling
        # the resulting shape is (B, D)
        valid_mask = ~mask # Not padded = valid entries
        summed = (enc_out * valid_mask.unsqueeze(-1)).sum(dim=1) # sets all padding positions to 0 then sums over the sequence dimension --> (batch_size, d_model)
        pooled = summed / torch.LongTensor(lengths)[:,None].to(enc_out) # divide with valid tokens get the mean

        # 7) apply a final linear layer to get the output of shape (B, output_dim)
        return self.output_layer(pooled)  



# ==== BUILD DATALOADERS ====
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn_transformer)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn_transformer)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn_transformer)


# ==== DEFINE MODEL ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TransformerEncoder(
    input_dim=3,       # (time, x, y)
    d_model=128,
    nhead=2,
    num_layers=2,
    dim_feedforward=256,
    output_dim=2       # (xpos, ypos)
).to(device)

criterion = nn.MSELoss()

# ==== TRAIN ====
epochs = 30
model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, device, epochs)

# ==== EVALUATE ====
test_loss = evaluate_model(model, test_loader, criterion, device)

# ==== PLOT RESULTS ====
plot_loss_curve(train_losses, val_losses, test_loss)
plot_predictions(model, test_loader, x_mean, x_std, y_mean, y_std, device)
plot_pull(model, test_loader, x_mean, x_std, y_mean, y_std, device)




""""
RMK: In gnn_trafo_helper.py we need to replace
    data, labels = data.to(device), labels.to(device)
with:
    data_vec, lengths = data       # unpack from list
    data_vec = data_vec.to(device) # move tensor to GPU
    labels = labels.to(device)     # move labels to GPU
    data = [data_vec, lengths]     # repackage data

since collate_fn_transformer returns a list of tensors which means we need to unpack the data
and repackage it again after moving each component to GPU
"""

