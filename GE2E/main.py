from data import get_training_data, load_AESDD_data, load_EMOVO_data
from dataset import GE2EDatset
from GE2Eloss import GE2ELoss
from sklearn.model_selection import train_test_split
from dataloader import LibriSamplesEval, LibriSamplesTest, GE2ELoader
from train import train, evaluate
from model import Wav2Vec2ClassificationHead
import torch
import wandb
import torch.nn.functional as F
import torch.optim as optim
from speechbrain.pretrained.interfaces import foreign_class
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)
classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")

if __name__ == "__main__":
    train_df = get_training_data()
    train_df, eval_df = train_test_split(train_df, test_size=0.2, random_state=11785, stratify=train_df["label"])
    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)
    train_df.to_csv(f"/content/train.csv", sep="\t", encoding="utf-8", index=False)
    eval_df.to_csv(f"/content/eval.csv", sep="\t", encoding="utf-8", index=False)

    EMOVO_label_dict = {'dis': 'Disgust', 'gio': 'Happiness', 'pau': 'Fear', 
                    'rab': 'Anger', 'sor': 'Surprise', 'tri': 'Sadness', 'neu': 'Neutral'}
    EMOVO_df = load_EMOVO_data(EMOVO_label_dict)
    AESDD_label_list = ["anger", "disgust", "fear", "happiness", "sadness"]
    AESDD_label_dict = {"anger": "Anger", "disgust": "Disgust", "fear": "Fear", "happiness": "Happiness", "sadness": "Sadness"}
    AESDD_df = load_AESDD_data(AESDD_label_list, AESDD_label_dict)
    test_dfs = [EMOVO_df[['path', 'label']], AESDD_df[['path', 'label']]]
    test_df = pd.concat(test_dfs)
    print("The combination of two test languages has size ", test_df.shape[0])
    test_df = test_df.sample(frac=1)
    test_df = test_df.reset_index(drop=True)
    print("Labels: ", test_df["label"].unique())
    test_df.rename(columns = {'path':'file'}, inplace = True)

    LABELS = ['Disgust', 'Sadness', 'Happiness', 'Fear', 'Neutral', 'Anger', 'Surprise']
    Label2id = {LABELS[i]: i for i in range(len(LABELS))}
    id2Label = {i: LABELS[i] for i in range(len(LABELS))}
    train_df = train_df.loc[train_df['label'].isin(LABELS)]


    batch_size = 32
    train_data = GE2EDatset(train_df, classifier, LABELS)
    val_data = LibriSamplesEval(eval_df, classifier, LABELS)
    test_data = LibriSamplesTest(test_df, classifier, LABELS)
    train_loader = GE2ELoader(train_df, classifier, LABELS, batch_size, num_steps=50)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)

    config = {
        "hidden_size": 768,
        "final_dropout": 0.2,
        "num_labels": 7,
        "lr": 0.002,
        "n_epoches": 50
    }
    best_score = 0
    model = Wav2Vec2ClassificationHead(config).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr']) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    train_criterion = GE2ELoss(init_w=10.0, init_b=-5.0, loss_method='softmax').to(device)
    eval_criterion = torch.nn.CrossEntropyLoss()
    wandb.watch(model, log='all')
    for epoch in range(config["n_epoches"]):
        train_loss = train(model, train_loader, optimizer, train_criterion, scheduler, epoch)
        eval_loss, eval_accuracy = evaluate(model, val_loader, eval_criterion, epoch)
        if best_score < eval_accuracy:
            best_score = eval_accuracy
            torch.save(model.state_dict(), "AESDD_model.pt")
            wandb.save('AESDD_model.pt')
        scheduler.step(eval_loss)
        wandb.log({
            "Epoch": epoch,
            "Train loss": train_loss,
            "Validation loss": eval_loss,
            "Validation accuracy": eval_accuracy,
            "Best Validation accuracy": best_score,
        })