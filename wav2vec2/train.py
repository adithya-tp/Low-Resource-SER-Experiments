import numpy as np
import torch
from transformers import EvalPrediction, TrainingArguments, AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import librosa
from sklearn.metrics import classification_report
from ann_gen import *
from create_dataset import create_data, EMOVO_process_dataset
from model_config import SpeechClassifierOutput, Wav2Vec2ClassificationHead, Wav2Vec2ForSpeechClassification
from train_model import DataCollatorCTCWithPadding
from CTCtrainer import CTCTrainer
import sys
sys.path.append("..")

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

def predict(batch):
    features = feature_extractor(batch["speech"], sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits 

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch

Wav2Vec2_PATH = "facebook/wav2vec2-base-100k-voxpopuli"
Wav2Vec2_pooling_mode = "mean"

if __name__ == "__main__":
    input_column = "path"
    output_column = "label"
    is_regression = False
    EMOVO_data_path = '../data/EMOVO/'
    AESDD_data_path = '../data/AESDD/'
    EMOVO_df = EMOVO_create_df(EMOVO_data_path)
    EMOVO_create_csv(EMOVO_df)
    EMOVO_train_dataset, EMOVO_val_dataset, EMOVO_test_dataset, EMOVO_label_list = create_data("EMOVO")

    config = AutoConfig.from_pretrained(
        Wav2Vec2_PATH,
        num_labels = len(EMOVO_label_list),
        label2id = {label: i for i, label in enumerate(EMOVO_label_list)},
        id2label = {i: label for i, label in enumerate(EMOVO_label_list)},
        finetuning_task="wav2vec2_clf"
    )
    setattr(config, 'pooling_mode', Wav2Vec2_pooling_mode)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(Wav2Vec2_PATH,)
    target_sampling_rate = feature_extractor.sampling_rate
    print(f"The target sampling rate: {target_sampling_rate}")

    #train_dataset, val_dataset, test_dataset = process_dataset(preprocess_function, train_dataset, val_dataset, test_dataset)
    EMOVO_train_dataset, EMOVO_val_dataset, EMOVO_test_dataset = EMOVO_process_dataset(EMOVO_label_list, feature_extractor, EMOVO_train_dataset, EMOVO_val_dataset, EMOVO_test_dataset, input_column, output_column)

    data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor, padding=True)

    is_regression = False

    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        Wav2Vec2_PATH,
        config=config,
    )
    print(model.config)
    model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir="TrainedModel/",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=10,
        fp16=True,
        save_steps=10,
        eval_steps=10,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
    )

    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=EMOVO_train_dataset,
        eval_dataset=EMOVO_val_dataset,
        tokenizer=feature_extractor,
    )

    trainer.train()
