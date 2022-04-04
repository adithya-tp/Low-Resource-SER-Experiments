import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio
from sklearn.model_selection import train_test_split
import os
import sys
import librosa
import IPython.display as ipd
# Loading the created dataset using datasets
from datasets import load_dataset, load_metric
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor


def create_data(ann_path):
    data_files = {
        "train": ann_path + "-train.csv", 
        "val": ann_path + "-val.csv",
        "test": ann_path + "-test.csv",
    }

    dataset = load_dataset("csv", data_files=data_files, delimiter="\t", )
    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    test_dataset = dataset["test"]
    print("train_dataset is ", train_dataset)
    print("val_dataset is ", val_dataset)
    print("test_dataset is ", test_dataset)

    input_column = "path"
    output_column = "label"

    # we need to distinguish the unique labels in our SER dataset
    label_list = train_dataset.unique(output_column)
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    print(f"A classification problem with {num_labels} classes: {label_list}")

    return train_dataset, val_dataset, test_dataset, label_list

# convert speech file to array and do resampling
def speech_file_to_array_fn(path, target_sampling_rate = 16000):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def label_to_id(label, label_list):
    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1
    return label

def preprocess_function(examples, feature_extractor, speech_file_to_array_fn, label_to_id, label, label_list, target_sampling_rate = 16000):
    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]
    result = feature_extractor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)
    return result

def EMOVO_process_dataset(label_list, feature_extractor, train_dataset, val_dataset, test_dataset, input_column, output_column):
    
    def EMOVO_speech_file_to_array_fn(path, target_sampling_rate = 16000):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        if len(speech.shape) > 1 and speech.shape[1] > 1:
            speech = np.mean(speech, axis=0)
        return speech

    def label_to_id(label, label_list):
        if len(label_list) > 0:
            return label_list.index(label) if label in label_list else -1
        return label
    
    def preprocess_function(examples, target_sampling_rate = 16000):
        speech_list = [EMOVO_speech_file_to_array_fn(path) for path in examples[input_column]]
        target_list = [label_to_id(label, label_list) for label in examples[output_column]]
        result = feature_extractor(speech_list, sampling_rate=target_sampling_rate)
        result["labels"] = list(target_list)
        return result
    
    train_dataset = train_dataset.map(
        preprocess_function,
        batch_size=10,
        batched=True,
        num_proc=4
    )
    val_dataset = val_dataset.map(
        preprocess_function,
        batch_size=10,
        batched=True,
        num_proc=4
    )

    test_dataset = test_dataset.map(
        preprocess_function,
        batch_size=10,
        batched=True,
        num_proc=4
    )

    return train_dataset, val_dataset, test_dataset
    
