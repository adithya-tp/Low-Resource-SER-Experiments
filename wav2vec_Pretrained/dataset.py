import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio
from sklearn.model_selection import train_test_split
import os
import torch

def load_AESDD_data(save_path, test_size, label_list):
    data = []
    for label in label_list:
        for path in tqdm(Path("/content/Acted Emotional Speech Dynamic Database/" + label + '/').glob("*.wav")):
            try:
                s = torchaudio.load(path)
                data.append({
                    "path": path,
                    "label": label
                })
            except Exception as e:
                pass
    df = pd.DataFrame(data)
    print(f"Step 0: {len(df)}")
    df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
    df = df.dropna(subset=["path"])
    df = df.drop("status", 1)
    print(f"Step 1: {len(df)}")
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    print("Labels: ", df["label"].unique())
    train_df, old_test_df = train_test_split(df, test_size=test_size, random_state=11785, stratify=df["label"])
    size = old_test_df.shape[0] // 2
    eval_df, test_df = old_test_df[:size], old_test_df[size:]
    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
    eval_df.to_csv(f"{save_path}/eval.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)
    return train_df, eval_df, test_df

def load_EMOVO_data(save_path, test_size):
    label_dict = {'dis': 'disguise', 'gio': 'joy', 'pau': 'fear', 'rab': 'anger', 'sor': 'surprise', 'tri': 'sadness', 'neu': 'neutral'}
    data = []
    folders = ['f1', 'f2', 'f3', 'm1', 'm2', 'm3']
    for folder in folders:
        for path in tqdm(Path("/content/EMOVO/" + folder + '/').glob("*.wav")):
            tokens = str(path).split('/')[-1].split('-')
            label = label_dict[tokens[0]]
            try:
                s = torchaudio.load(path)
                data.append({
                    "path": path,
                    "label": label
                })
            except Exception as e:
                pass
    df = pd.DataFrame(data)
    print(f"Step 0: {len(df)}")
    df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
    df = df.dropna(subset=["path"])
    df = df.drop("status", 1)
    print(f"Step 1: {len(df)}")
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    print("Labels: ", df["label"].unique())
    train_df, old_test_df = train_test_split(df, test_size=test_size, random_state=11785, stratify=df["label"])
    size = old_test_df.shape[0] // 2
    eval_df, test_df = old_test_df[:size], old_test_df[size:]
    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
    eval_df.to_csv(f"{save_path}/eval.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)
    return train_df, eval_df, test_df

class LibriSamples(torch.utils.data.Dataset):
    def __init__(self, dataframe, model, label_list): # You can use partition to specify train or dev
        self.X_paths = dataframe["path"].tolist()
        self.labels = dataframe["label"].tolist()
        self.model = model
        self.label_list = label_list
        assert(len(self.X_paths) == len(self.labels))
        pass

    def __len__(self):
        return len(self.X_paths)

    def __getitem__(self, ind):
        X_data, _ = torchaudio.load(self.X_paths[ind])
        if X_data.shape[0] > 1:
            X_data = torch.mean(X_data, axis=0)
        embeddings = self.model.encode_batch(X_data)
        #Yy = torch.tensor(self.label_list.index(self.labels[ind]))
        Yy = self.label_list.index(self.labels[ind])
        #print("xx has shape ", embeddings.shape)
        #print("yy has shape ", Yy.shape)
        return embeddings, Yy