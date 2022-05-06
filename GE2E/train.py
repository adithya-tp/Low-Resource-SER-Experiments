from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)


def train(model, train_loader, optimizer, criterion, scheduler, epoch):
    torch.cuda.empty_cache()
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader)
    count = 0
    for i, data in enumerate(pbar):
        count += 1
        data = data.float().to(device)
        optimizer.zero_grad()
        output = model(data, return_feats=True)
        loss = criterion(output)
        loss.backward()
        total_loss += loss.item()
        pbar.set_postfix({"loss": total_loss / count})
        optimizer.step()
    avg_loss = total_loss /count
    print('Train Epoch: {} \tLoss: {:.6f} \t lr: {:.4f}'.format(
                    epoch, avg_loss, float(optimizer.param_groups[0]['lr'])))
    return avg_loss

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
            output = model(data, return_feats=False)
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