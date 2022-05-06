import torchaudio
import tqdm
import pandas as pd
from pathlib import Path

def load_AESDD_data(label_list, label_dict):
    data = []
    for label in label_list:
        for path in tqdm(Path("/content/Acted Emotional Speech Dynamic Database/" + label + '/').glob("*.wav")):
            try:
                s = torchaudio.load(path)
                data.append({
                    "path": path,
                    "label": label_dict[label]
                })
            except Exception as e:
                pass
    df = pd.DataFrame(data)
    print(f"Step 0 AESDD has data: {len(df)}")
    df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
    df = df.dropna(subset=["path"])
    df = df.drop("status", 1)
    print(f"Step 1 AESDD has data: {len(df)}")
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    print("AESDD has Labels: ", df["label"].unique())
    return df


def load_EMOVO_data(label_dict):
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
    print(f"Step 0 EMOVO has data : {len(df)}")
    df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
    df = df.dropna(subset=["path"])
    df = df.drop("status", 1)
    print(f"Step 1 EMOVO has data : {len(df)}")
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    print("EMOVO has Labels: ", df["label"].unique())
    return df

def get_training_data():
    French_label_dict = {'Peur': 'Fear', "Joie": "Happiness", "Neutre": "Neutral", "Tristesse": "Sadness", 
                     "Surprise": "Surprise", "ColКre": "Anger", "DВgoЦt": "Disgust"}
    French_df = pd.read_csv("/content/drive/MyDrive/11785/Project/Dataset/French.csv")  
    print("The size of French data is ", French_df.shape[0])
    English_dict = {'sad': "Sadness", 'hap': "Happiness", 'fru': "Frustration", 
                    'exc': "Excitement", 'ang': "Anger", 'xxx': "Unknown", 
                    'neu': "Neutral", 'sur': "Surprise", 'oth': "Other",  'fea': "Fear", 'dis': "Disgust"}
    English_df = pd.read_csv("/content/drive/MyDrive/11785/Project/Dataset/df_iemocap.csv")
    English_df["label"] = English_df["emotion"].map(English_dict)
    English_df.rename(columns = {'wav_file':'file'}, inplace = True)
    del English_df["emotion"]
    print("The size of English data is ", English_df.shape[0])
    German_dict = {'afraid': "Fear", 'angry': "Anger", 'boring': "Bore", 
                    'disgust': "Disgust", 'happy': "Happiness", 'neutral': "Neutral", 'sad': "Sadness"}
    German_df = pd.read_csv("/content/drive/MyDrive/11785/Project/Dataset/German_df.csv")  
    German_df["label"] = German_df["emotion_en"].map(German_dict)
    del German_df["emotion_en"]
    German_df.rename(columns = {'uuid':'file'}, inplace = True)
    German_df['file'] = German_df['file'].map(lambda x: "/content/German/wav/" + x + ".wav")
    print("The size of German data is ", German_df.shape[0])
    #train_dfs = [English_df[['file', 'label']], French_df[['file', 'label']], German_df[['file', 'label']]]
    train_dfs = [French_df[['file', 'label']], German_df[['file', 'label']]]
    train_df = pd.concat(train_dfs)
    print("The combination of three languages has size ", train_df.shape[0])
    train_df = train_df.sample(frac=1)
    train_df = train_df.reset_index(drop=True)
    print("Labels: ", train_df["label"].unique())