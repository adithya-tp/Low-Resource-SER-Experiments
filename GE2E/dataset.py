import torchaudio
import torch
import random


class GE2EDatset(object):

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

    def sample(self, batch_size):
        output = []
        for label in self.label_list:
            utt_sets = self.emotion_dict[label]
            if len(utt_sets) < batch_size:
                raise RuntimeError("Emotion {} can not got enough utterance with M = {:d}".format(label, batch_size))
            samp_utts = random.sample(utt_sets, batch_size)
            assert(len(samp_utts) == batch_size)
            embeds = []
            for path in samp_utts:
                X_data, sampling_rate = torchaudio.load(path)
                resampler = torchaudio.transforms.Resample(sampling_rate, 1600)
                X_data = resampler(X_data).squeeze()
                if len(X_data.shape) > 1 and X_data.shape[0] > 1:
                    X_data = torch.mean(X_data, axis=0)
                embeddings = self.model.encode_batch(X_data)
                embeds.append(embeddings)
            emotion_chunks = torch.cat(embeds, dim=0)
            output.append(emotion_chunks)
        output = torch.stack(output, dim=0)
        return output