from torch.utils.data import DataLoader
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
import os

import constants.EMOTIONS as EMOTS
from audio_datasets.aessd_dataset import AESDD_Greek
import constants.PATHS as PATHS

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

    data_loader = DataLoader(
        dataset=AESDD_Greek(
            BASE_DIR=PATHS.AESDD_DIR,
            ANNOTATIONS_PATH=os.path.join(PATHS.ROOT_DIR, 'annotations/annotations_aesdd.csv'),
            target_sampling_rate=feature_extractor.sampling_rate
        ),
        batch_size=2,
        shuffle=True,
        collate_fn=AESDD_Greek.collate_fn
    )

    (x, lx, y) = next(iter(data_loader))
    result = feature_extractor(x, sampling_rate=feature_extractor.sampling_rate)
    print(f"Batch of x: {x}, \n Lengths of x: {lx}")
    print(f"Batch of y: {y}")

if __name__ == '__main__':
    main()
