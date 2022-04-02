import constants.EMOTIONS as EMOTS
from audio_datasets.aessd_dataset import AESDD_Greek

from transformers import AutoConfig, Wav2Vec2FeatureExtractor

def get_wavenet2_config(HUGGING_FACE_MODEL_PATH):
    emotions = EMOTS.AESDD_EMOTIONS
    conf = AutoConfig.from_pretrained(
        HUGGING_FACE_MODEL_PATH,
        num_labels = len(emotions),
        label2id=dict(zip( emotions, range(len(emotions)) )),
        id2label=dict(zip( range(len(emotions)) , emotions )),
        finetuning_task="wav2vec2_clf",
    )
    setattr(conf, 'pooling_mode', 'mean')
    return conf

def main():
    model_name_or_path = "facebook/wav2vec2-base-100k-voxpopuli"
    pooling_mode = "mean"
    config = get_wavenet2_config(
        model_name_or_path
    )

    feature_extractor = \
        Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path,)
    target_sampling_rate = feature_extractor.sampling_rate

    data_loader = 

if __name__ == '__main__':
    main()
