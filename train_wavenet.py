from transformers import AutoConfig, Wav2Vec2FeatureExtractor

import constants.EMOTIONS as EMOTS
from train_utils.loaders import get_data_loaders
from train_utils.trainer import train_setup, train

def get_wavenet2_config(HUGGING_FACE_MODEL_PATH):
    emotions = EMOTS.AESDD_EMOTIONS
    conf = AutoConfig.from_pretrained(
        HUGGING_FACE_MODEL_PATH,
        num_labels = len(emotions),
        label2id=dict(zip( emotions, range(len(emotions)) )),
        id2label=dict(zip( range(len(emotions)) , emotions )),
        finetuning_task="wav2vec2_clf",
    )
    return conf

def main():
    model_name_or_path = "facebook/wav2vec2-base-100k-voxpopuli"
    config = get_wavenet2_config(
        HUGGING_FACE_MODEL_PATH=model_name_or_path,
    )

    feature_extractor = \
        Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path,)

    train_loader, val_loader = get_data_loaders(
        train_annotations_path='annotations/annotations_aesdd.csv',
        val_annotations_path='annotations/annotations_aesdd.csv',
        feature_extractor=feature_extractor
    )
    model, criterion, optimizer, scaler = train_setup(model_name_or_path, config)
    train(train_loader, val_loader, model, criterion, optimizer, scaler)

    # (x, y) = next(iter(train_loader))
    # print(f"Batch of x: {x}")
    # print(f"Batch of y: {y}")

if __name__ == '__main__':
    main()
