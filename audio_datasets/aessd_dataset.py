from torch.utils.data import Dataset
import pandas as pd
import os
import torchaudio
import torch

import sys
sys.path.append(".")
sys.path.append("..")

class AESDD_Greek(Dataset):
    """
    Pytorch Dataset for the Acted Emotional Speech 
    Dynamic Database (AESDD) speech emotion dataset
    """
    def __init__(self, BASE_DIR, ANNOTATIONS_PATH, feature_extractor):
        self.base_dir = BASE_DIR
        self.annotations = pd.read_csv(ANNOTATIONS_PATH)
        self.feature_extractor = feature_extractor
        self.emotions = ["fear", "sadness", "happiness", "anger", "disgust"]

    def _fetch_audio(self, index, folder):
        file_name = self.annotations.loc[index]['file_path']
        audio_sample_path = os.path.join(self.base_dir, folder, file_name)
        audio, sampling_rate = torchaudio.load(audio_sample_path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.feature_extractor.sampling_rate)
        resampled_audio = resampler(audio)
        return resampled_audio

    def _fetch_label(self, label):
        return self.emotions.index(label)
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label = self.annotations.loc[index]['folder_or_label']
        label_idx = self._fetch_label(label)
        audio = self._fetch_audio(index, folder=label)
        return audio, label_idx


class Wav2Vec2Collator(object):
    """
    Custom Collator to accept parameters.
    Passes batch of data into Wav2Vec feature extractor 
    and returns padded tensor of values as Pytorch Tensor
    """
    def __init__(self, *params):
        self.params = params
    
    def __call__(self, batch):
        feature_extractor = self.params[0]
        batch_x = [x.numpy().squeeze() for (x, _) in batch]
        sr = feature_extractor.sampling_rate
        batch_x_pad = feature_extractor(batch_x, sampling_rate=sr, padding=True, return_tensors='pt')

        batch_y = [y for (_, y) in batch]
        return batch_x_pad['input_values'], torch.tensor(batch_y)