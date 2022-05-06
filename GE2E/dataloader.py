from dataset import GE2EDatset
import torch
import torchaudio

class GE2ELoader(object):
    def __init__(self, dataframe, model, label_list, batch_size, num_steps=100):
        self.GE2EDatset = GE2EDatset(dataframe, model, label_list)
        self.batch_size = batch_size
        self.num_steps = num_steps

    def _sample(self):
        return self.GE2EDatset.sample(self.batch_size)

    def __iter__(self):
        for _ in range(self.num_steps):
            yield self._sample()


class LibriSamplesEval(torch.utils.data.Dataset):
    def __init__(self, dataframe, model, label_list):

        # TODO
        self.X_paths = dataframe["file"].tolist()
        self.labels = dataframe["label"].tolist()
        self.emotion_dict = {}
        for path, label in zip(self.X_paths, self.labels):
            if label not in self.emotion_dict:
                self.emotion_dict[label] = []
            self.emotion_dict[label].append(path)
        self.model = model
        self.label_list = label_list
        assert(len(self.X_paths) == len(self.labels))
        pass

    def __len__(self):
        # TODO
        return len(self.X_paths)

    def __getitem__(self, ind):
        path = self.X_paths[ind]
        '''
        if '/content' not in path:
            speech_array, sampling_rate = self.audio_dict[path] 
            speech_array = torch.from_numpy(speech_array)
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            X_data = resampler(speech_array).squeeze()
            print("english size is ", x_data.shape)
        else:
        '''
        X_data, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, 1600)
        X_data = resampler(X_data).squeeze()
        #print("no english size is ", x_data.shape)
        #X_data, _ = torchaudio.load(self.X_paths[ind])
        if len(X_data.shape) > 1 and X_data.shape[0] > 1:
            X_data = torch.mean(X_data, axis=0)
        embeddings = self.model.encode_batch(X_data)
        Yy = self.label_list.index(self.labels[ind])
        return embeddings, Yy

class LibriSamplesTest(torch.utils.data.Dataset):
    def __init__(self, dataframe, model, label_list): # You can use partition to specify train or dev
        self.X_paths = dataframe["file"].tolist()
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
        Yy = self.label_list.index(self.labels[ind])
        return embeddings, Yy