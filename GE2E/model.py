import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["final_dropout"])
        self.out_proj = nn.Linear(config["hidden_size"], config["num_labels"])

    def forward(self, features, return_feats, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        if return_feats:
           return x
        x = self.out_proj(x)
        return x