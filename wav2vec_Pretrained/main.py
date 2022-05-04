import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import sys
from datasets import load_dataset, load_metric
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2Model, TrainingArguments
from speechbrain.pretrained.interfaces import foreign_class
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.file_utils import ModelOutput
import wandb
from dataset import load_AESDD_data, load_EMOVO_data, LibriSamples
from model import Wav2Vec2ClassificationHead, train, evaluate


if __name__ == "__main__":
    args = {
        "test_data": "EMOVO",
        "hidden_size": 768,
        "final_dropout": 0.2,
        "num_labels": 7,
        "lr": 0.002,
        "n_epoches": 50
    }

    save_path = "/content"
    test_size = 0.3
    if args["test_data"] == "EMOVO":
        train_df, eval_df, test_df = load_EMOVO_data(save_path, test_size)
    else:
        train_df, eval_df, test_df = load_AESDD_data(save_path, test_size)
    data_files = {
        "train": "/content/train.csv", 
        "validation": "/content/eval.csv",
        "test": "/content/test.csv",
    }
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    print(train_dataset)
    print(eval_dataset)
    print(test_dataset)
    input_column = "path"
    output_column = "label"
    train_label_list = train_dataset.unique(output_column)
    train_label_list.sort()  # Let's sort it for determinism
    train_num_labels = len(train_label_list)
    print(f"For training, A classification problem with {train_num_labels} classes: {train_label_list}")
    eval_label_list = eval_dataset.unique(output_column)
    eval_label_list.sort()  # Let's sort it for determinism
    eval_num_labels = len(eval_label_list)
    print(f"For validation, A classification problem with {eval_num_labels} classes: {eval_label_list}")
    test_label_list = test_dataset.unique(output_column)
    test_label_list.sort()  # Let's sort it for determinism
    test_num_labels = len(test_label_list)
    print(f"For test, A classification problem with {test_num_labels} classes: {test_label_list}")
    print(train_label_list)
    print(eval_label_list)
    print(test_label_list)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)
    batch_size = 32
    classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
    train_data = LibriSamples(train_df, classifier, train_label_list)
    val_data = LibriSamples(eval_df, classifier, train_label_list)
    test_data = LibriSamples(test_df, classifier, train_label_list)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)

    best_score = 0
    model = Wav2Vec2ClassificationHead(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr']) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    criterion = torch.nn.CrossEntropyLoss()
    wandb.watch(model, log='all')
    for epoch in range(args["n_epoches"]):
        train_loss = train(model, train_loader, optimizer, criterion, scheduler, epoch)
        eval_loss, eval_accuracy = evaluate(model, val_loader, criterion, epoch)
        if best_score < eval_accuracy:
            best_score = eval_accuracy
            torch.save(model.state_dict(), "best_model.pt")
            wandb.save('best_model.pt')
        scheduler.step(eval_loss)
        wandb.log({
            "Epoch": epoch,
            "Train loss": train_loss,
            "Validation loss": eval_loss,
            "Validation accuracy": eval_accuracy,
            "Best Validation accuracy": best_score,
        })