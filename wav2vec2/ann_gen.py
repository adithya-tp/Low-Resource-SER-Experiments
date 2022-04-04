import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio
from sklearn.model_selection import train_test_split
import os
import sys
sys.path.append("..")
import librosa
import IPython.display as ipd


def EMOVO_create_df(data_path):
    data = []
    folders = ['f1', 'f2', 'f3', 'm1', 'm2', 'm3']
    label_dict = {'dis': 'disguise', 'gio': 'joy', 'pau': 'fear', 'rab': 'anger', 'sor': 'surprise', 'tri': 'sadness', 'neu': 'neutral'}
    for folder in folders:
        for path in tqdm(Path(data_path + folder + '/').glob("*.wav")):
            tokens = str(path).split('/')[-1].split('-')
            label = label_dict[tokens[0]]
            
            try:
                # There are some broken files
                s = torchaudio.load(path)
                data.append({
                    "path": path,
                    "label": label
                })
            except Exception as e:
                pass

    df = pd.DataFrame(data)
    # Filter broken and non-existed paths
    print(df[:5])
    print(f"Step 0: {len(df)}")
    df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
    df = df.dropna(subset=["path"])
    df = df.drop("status", 1)
    print(f"Step 1: {len(df)}")

    # sample data
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)

    # # print dataset status
    print("Labels: ", df["label"].unique())
    return df


def create_df(data_path = '/content/data/aesdd'):
    data = []

    for path in tqdm(Path(data_path).glob("**/*.wav")):
        name = str(path).split('/')[-1].split('.')[0]
        label = str(path).split('/')[-2]
        
        try:
            # There are some broken files
            s = torchaudio.load(path)
            data.append({
                # "name": name,
                "path": path,
                "label": label
            })
        except Exception as e:
            # print(str(path), e)
            pass

    df = pd.DataFrame(data)
    # Filter broken and non-existed paths

    print(f"Step 0: {len(df)}")
    df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
    df = df.dropna(subset=["path"])
    df = df.drop("status", 1)
    print(f"Step 1: {len(df)}")

    # sample data
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)

    # # print dataset status
    print("Labels: ", df["label"].unique())
    print()
    df.groupby("label").count()[["path"]]
    return df


def EMOVO_create_csv(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=101, stratify=df["label"])
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=101, stratify=test_df["label"])
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df.to_csv(f"EMOVO-train.csv", sep="\t", encoding="utf-8", index=False)
    val_df.to_csv(f"EMOVO-val.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"EMOVO-test.csv", sep="\t", encoding="utf-8", index=False)
    print("The training df has shape ", train_df.shape)
    print("The validation df has shape ", val_df.shape)
    print("The test df has shape ", test_df.shape)

def create_csv(df, save_path = "/content/data"):

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=101, stratify=df["label"])
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=101, stratify=test_df["label"])

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
    val_df.to_csv(f"{save_path}/val.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)

    print(train_df.shape)
    print(val_df.shape)
    print(test_df.shape)


def play_audio(df, idx):
  sample = df.iloc[idx]
  path = sample["path"]
  label = sample["label"]


  print(f"ID Location: {idx}")
  print(f"      Label: {label}")
  print()

  speech, sr = torchaudio.load(path)
  speech = speech[0].numpy().squeeze()
  speech = librosa.resample(np.asarray(speech), sr, 16_000)
  return speech