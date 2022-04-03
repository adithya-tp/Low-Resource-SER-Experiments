from tqdm import tqdm
import torch
import torch.nn as nn

import constants.HYPER_PARAMETERS as HP
from models.wav2vec2 import Wav2vec2SER as Network
from train_utils.val_acc import get_validation_acc

def train_setup(model_name_or_path, config):
    model = Network.from_pretrained(
        model_name_or_path,
        config=config
    )
    model.feat_extractor.feature_extractor._freeze_parameters()
    model.cuda()
    num_trainable_parameters = 0
    for p in model.parameters():
        num_trainable_parameters += p.numel()
    print("Number of Params: {}".format(num_trainable_parameters))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=HP.LR)
    scaler = torch.cuda.amp.GradScaler()
    return (model, criterion, optimizer, scaler)

def train(train_loader, val_loader, model, criterion, optimizer, scaler):
    for epoch in range(HP.EPOCHS):
        model.train()
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, desc='Train', leave=False, position=0)
        num_correct = 0
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(x)
                loss = criterion(outputs, y)
            
            num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
            total_loss += float(loss)

            batch_bar.set_postfix(
                    acc="{:.04f}%".format(100 * num_correct / ((i + 1) * HP.BATCH_SIZE)),
                    loss="{:.04f}".format(float(total_loss / (i + 1))),
                    num_correct=num_correct,
                    lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
            )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_bar.update()
        batch_bar.close()

        train_acc = 100 * num_correct / len(train_loader.dataset)
        train_loss = float(total_loss / len(train_loader))
        # val_acc, val_loss = get_validation_acc(model, val_loader, criterion)

        # print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Validation Acc {:.04f}, Validation Loss {:.04f}, Learning Rate {:.04f}".format(
        #     epoch + 1,
        #     HP.EPOCHS,
        #     train_acc,
        #     train_loss,
        #     val_acc,
        #     val_loss,
        #     float(optimizer.param_groups[0]['lr'])
        #     )
        # )

        print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Learning Rate {:.04f}".format(
            epoch + 1,
            HP.EPOCHS,
            train_acc,
            train_loss,
            float(optimizer.param_groups[0]['lr'])
            )
        )