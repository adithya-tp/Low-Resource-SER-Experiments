from train import *
import torch
import os
import sys
from datasets import load_dataset, load_metric
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
test_dataset = load_dataset("csv", data_files={"test": "EMOVO-test.csv"}, delimiter="\t")["test"]
output_column = "label"
EMOVO_label_list = test_dataset.unique(output_column)
EMOVO_label_list.sort()
print(EMOVO_label_list)
model_name_or_path = "TrainedModel/checkpoint-590/"

#config = AutoConfig.from_pretrained(model_name_or_path)
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=len(EMOVO_label_list),
    label2id={label: i for i, label in enumerate(EMOVO_label_list)},
    id2label={i: label for i, label in enumerate(EMOVO_label_list)},
    local_files_only=True
)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path, local_files_only=True)
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=len(EMOVO_label_list),
    label2id={label: i for i, label in enumerate(EMOVO_label_list)},
    id2label={i: label for i, label in enumerate(EMOVO_label_list)},
)
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path, id2label={i: label for i, label in enumerate(EMOVO_label_list)}, local_files_only=True).to(device)

def EMOVO_test_speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    #print("loaded")
    speech_array = speech_array.squeeze().numpy()
    if len(speech_array.shape) > 1 and speech_array.shape[1] > 1:
        speech_array = np.mean(speech_array, axis=0)
    #print("squeezed")
    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, feature_extractor.sampling_rate)

    batch["speech"] = speech_array
    return batch

def predict(batch):
    features = feature_extractor(batch["speech"], sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits 

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch


if __name__ == "__main__":
    test_dataset = test_dataset.map(EMOVO_test_speech_file_to_array_fn)


    result = test_dataset.map(predict, batched=True, batch_size=8)
    label_names = [config.id2label[i] for i in range(config.num_labels)]
    y_true = [config.label2id[name] for name in result["label"]]
    y_pred = result["predicted"]
    print(classification_report(y_true, y_pred, target_names=label_names))