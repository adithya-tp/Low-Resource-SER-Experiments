import torch
from tqdm import tqdm
import constants.HYPER_PARAMETERS as HP

def get_validation_acc(model, val_loader, criterion):
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')
    num_correct = 0
    val_loader_length = len(val_loader.dataset)
    total_val_loss = 0
    for i, (x, y) in enumerate(val_loader):
        x = x.cuda()

        with torch.no_grad():
            outputs = model(x)
            loss = criterion(outputs, y)
            
        total_val_loss += loss

        num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
        batch_bar.set_postfix(acc="{:.04f}%".format(100 * num_correct / ((i + 1) * HP.BATCH_SIZE)))

        batch_bar.update()

    batch_bar.close()
    print("Validation: {:.04f}%".format(100 * num_correct / val_loader_length))
    return ( 100 * num_correct / val_loader_length, float(total_val_loss / len(val_loader)) )