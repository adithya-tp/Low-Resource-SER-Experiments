import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import torchaudio

import sys
sys.path.append(".")
sys.path.append("..")

class AESDD_Greek(Dataset):
    """
    Pytorch Dataset for the Acted Emotional Speech 
    Dynamic Database (AESDD) speech emotion dataset
    """
    def __init__(self, BASE_DIR, ANNOTATIONS_PATH, target_sampling_rate):
        self.base_dir = BASE_DIR
        self.annotations = pd.read_csv(ANNOTATIONS_PATH)
        self.target_sampling_rate = target_sampling_rate
        self.emotions = ["fear", "sadness", "happiness", "anger", "disgust"]

    def _fetch_audio(self, index, folder):
        file_name = self.annotations.loc[index]['file_path']
        audio_sample_path = os.path.join(self.base_dir, folder, file_name)
        audio, sampling_rate = torchaudio.load(audio_sample_path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
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
    
    # def collate_fn(batch):
    #     batch_x = [x for x,y in batch]
    #     batch_y = [y for x,y in batch]

    #     batch_x_pad = pad_sequence(batch_x, batch_first=True) # TODO: pad the sequence with pad_sequence (already imported)
    #     lengths_x = [len(x) for x in batch_x] # TODO: Get original lengths of the sequence before padding

    #     batch_y_pad = pad_sequence(batch_y, batch_first=True) # TODO: pad the sequence with pad_sequence (already imported)
    #     lengths_y = [len(y) for y in batch_y] # TODO: Get original lengths of the sequence before padding
    #     return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)

def test():
    import constants.PATHS as PATHS
    dataset = AESDD_Greek(
        BASE_DIR=PATHS.AESDD_DIR,
        ANNOTATIONS_PATH=os.path.join(PATHS.AESDD_DIR, 'annotations/annotations.csv')
    )
    print(len(dataset)) # Should be 605
    print(dataset[0]) # Print first data point
    print(dataset[0][0].shape, dataset[1][0].shape) # audio has different lengths
    
if __name__ == '__main__':
    test()