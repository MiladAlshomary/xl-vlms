import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- heads and decoder (simpler since encoder is frozen) ---
class SplitHeads(nn.Module):
    def __init__(self, z_dim, rc=128, rs=64):
        super().__init__()
        self.u_head = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(),
            nn.Linear(128, rc)
        )
        self.v_head = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(),
            nn.Linear(128, rs)
        )
    def forward(self, z):
        return self.u_head(z), self.v_head(z)

class Decoder(nn.Module):
    def __init__(self, rc, rs, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rc+rs, 256), nn.ReLU(),
            nn.Linear(256, z_dim)
        )
    def forward(self, uv):
        return self.net(uv)

class StyleClassifier(nn.Module):
    def __init__(self, rs, num_styles):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rs, 128), nn.ReLU(),
            nn.Linear(128, num_styles)
        )
    def forward(self, v):
        return self.net(v)

def hsic_linear(X, Y, eps=1e-6):
    B = X.size(0)
    if B <= 1:
        return (X.sum() * 0.0)
    # normalize each feature dimension
    X = (X - X.mean(0)) / (X.std(0) + eps)
    Y = (Y - Y.mean(0)) / (Y.std(0) + eps)
    C = X.t() @ Y
    return (C**2).sum() / ((B - 1.0) ** 2)


def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)



# --- Training loop with stability fixes ---
def train_latent_disentanglement(z_data, style_labels, rc=128, rs=64,
                                 beta_hsic=1e-3, gamma_mi=0.1, epochs=20, batch_size=64):

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    int_labels = le.fit_transform(style_labels)
    style_labels = torch.from_numpy(int_labels).long()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = TensorDataset(z_data.clone().detach().float().to(device),
                       style_labels.to(device))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    z_dim = z_data.size(1)
    num_styles = int(style_labels.max()) + 1

    heads = SplitHeads(z_dim, rc, rs).to(device)
    decoder = Decoder(rc, rs, z_dim).to(device)
    style_clf = StyleClassifier(rs, num_styles).to(device)

    # init weights
    heads.apply(init_weights_xavier)
    decoder.apply(init_weights_xavier)
    style_clf.apply(init_weights_xavier)

    opt = torch.optim.Adam(
        list(heads.parameters()) + list(decoder.parameters()) + list(style_clf.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    ce_loss = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    for ep in range(epochs):
        tot_recon, tot_hsic, tot_clf = 0, 0, 0
        for z, y in dl:
            z, y = z.to(device), y.to(device)

            U, V = heads(z)
            z_hat = decoder(torch.cat([U, V], 1))
            recon = mse(z_hat, z)

            hsic = hsic_linear(U, V)
            logits = style_clf(V)
            clf = ce_loss(logits, y)

            loss = recon + beta_hsic * hsic - gamma_mi * clf

            opt.zero_grad()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(heads.parameters()) + list(decoder.parameters()) + list(style_clf.parameters()), 
                max_norm=5.0
            )
            opt.step()

            tot_recon += recon.item()
            tot_hsic += hsic.item()
            tot_clf += clf.item()

        print(f"Epoch {ep+1}/{epochs} | Recon {tot_recon/len(dl):.4f} | "
              f"HSIC {tot_hsic/len(dl):.6f} | CE {tot_clf/len(dl):.4f}")

    return heads, decoder, style_clf