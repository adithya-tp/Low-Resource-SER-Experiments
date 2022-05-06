from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["final_dropout"])
        self.out_proj = nn.Linear(config["hidden_size"], config["num_labels"])

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

def evaluate(model, val_loader, criterion, epoch):
    torch.cuda.empty_cache()
    model.eval()
    total_loss = 0
    pred_y_list = []
    true_y_list = []
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(val_loader, position = 0)):
            data = data.float().to(device)
            target = target.long().to(device)
            output = model(data)
            output = torch.squeeze(output, dim=1)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred_y = torch.argmax(output, axis=1)
            pred_y_list.extend(pred_y.tolist())
            true_y_list.extend(target.tolist())
    avg_loss = total_loss / len(val_loader)
    val_accuracy =  accuracy_score(true_y_list, pred_y_list)
    print('Validation Epoch: {} \tLoss: {:.6f} \tAccuracy: {:.6f}'.format(
                    epoch, avg_loss, val_accuracy))
    return avg_loss, val_accuracy

def train(model, train_loader, optimizer, criterion, scheduler, epoch):
    torch.cuda.empty_cache()
    model.train()
    total_loss = 0
    for i, (data, target) in enumerate(tqdm(train_loader, position = 0)):
        data = data.float().to(device)
        target = target.long().to(device)
        optimizer.zero_grad()
        output = model(data)
        output = torch.squeeze(output, dim=1)
        #print(output.shape)
        #print(target.shape)
        loss = criterion(output, target)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    avg_loss = total_loss / len(train_loader)
    print('Train Epoch: {} \tLoss: {:.6f} \t lr: {:.4f}'.format(
                    epoch, avg_loss, float(optimizer.param_groups[0]['lr'])))
    return avg_loss