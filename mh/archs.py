"""Private architectures for the heterogeneous scenario (MobiHet_IDS.md §4.3, balanced 25/25/25/25; TCN replaced by
DAGSNet, the protocol model). Ported from archive3 Build20 `private_model`; each returns (logits, latent) like DAGSNet."""
import torch.nn as nn

D, C = 66, 16


class CNN1d(nn.Module):
    """Convolution across the 66 feature positions of one message."""
    def __init__(self, w=256):
        super().__init__()
        self.body = nn.Sequential(nn.Unflatten(1, (1, D)), nn.Conv1d(1, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU(),
                                  nn.Conv1d(16, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU(), nn.Flatten(),
                                  nn.Linear(16 * D, w), nn.ReLU())
        self.out = nn.Linear(w, C)

    def forward(self, x):
        h = self.body(x)
        return self.out(h), h


class GRUNet(nn.Module):
    """GRU over a length-1 sequence (single message), as the archive registry ran it."""
    def __init__(self, w=256):
        super().__init__()
        self.gru = nn.GRU(D, w, batch_first=True); self.out = nn.Linear(w, C)

    def forward(self, x):
        h = self.gru(x.unsqueeze(1))[0][:, -1]
        return self.out(h), h
