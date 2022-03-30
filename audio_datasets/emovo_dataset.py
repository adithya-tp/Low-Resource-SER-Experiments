from torch.utils.data import Dataset
import pandas as pd
import os
import torchaudio

import sys
sys.path.append(".")
sys.path.append("..")

class EMOVO_Italian(Dataset):
    """
    Pytorch Dataset for EMOVO speech emotion dataset
    """
    def __init__(self, BASE_DIR, ANNOTATIONS_PATH):
        self.base_dir = BASE_DIR
        self.annotations = pd.read_csv(ANNOTATIONS_PATH)

    def _get_audio_sample_path(self, index):
        folder = self.annotations.iloc[index]['folder']
        file_name = self.annotations.iloc[index]['file_path']
        return os.path.join(self.base_dir, folder, file_name)

    def _get_label(self, index):
        label = self.annotations.iloc[index]['emotion']
        return label
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        audio, _ = torchaudio.load(audio_sample_path)

        label = self._get_label(index)
        return audio, label

# def test():
#     import constants.PATHS as PATHS
#     dataset = EMOVO_Italian(
#         BASE_DIR=PATHS.EMOVO_DIR,
#         ANNOTATIONS_PATH=os.path.join(PATHS.EMOVO_DIR, 'annotations/annotations.csv')
#     )
#     print(len(dataset)) # Should be 588
#     print(dataset[0]) # Print first data point
#     print(dataset[0][0].shape, dataset[1][0].shape) # audio has different lengths
    
# if __name__ == '__main__':
#     test()