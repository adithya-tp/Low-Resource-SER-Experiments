from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torchaudio

import sys
sys.path.append(".")
sys.path.append("..")
import constants.PATHS as PATHS

class AESDD_Greek(Dataset):
    """
    Pytorch Dataset for the Acted Emotional Speech Dynamic Database (AESDD) dataset
    """
    def __init__(self, BASE_DIR, ANNOTATIONS_PATH):
        self.base_dir = BASE_DIR
        self.annotations = pd.read_csv(ANNOTATIONS_PATH, skiprows=1, header=None)

    def _get_audio_sample_path(self, index, folder):
        file_name = self.annotations.iloc[index][0]
        return os.path.join(self.base_dir, folder, file_name)
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label = self.annotations.iloc[index][1]
        audio_sample_path = self._get_audio_sample_path(index, folder=label)
        audio, _ = torchaudio.load(audio_sample_path)
        return audio, label

# def test():
#     dataset = AESDD_Greek(
#         BASE_DIR=PATHS.AESDD_DIR,
#         ANNOTATIONS_PATH=os.path.join(PATHS.AESDD_DIR, 'annotations/annotations.csv')
#     )
#     print(len(dataset)) # Should be 605
#     print(dataset[0]) # Print first data point
#     print(dataset[0][0].shape, dataset[1][0].shape) # audio has different lengths
    
# if __name__ == '__main__':
#     test()