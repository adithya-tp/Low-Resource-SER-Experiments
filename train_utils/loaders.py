from torch.utils.data import DataLoader
import os

from audio_datasets.aessd_dataset import AESDD_Greek
import constants.PATHS as PATHS
import constants.HYPER_PARAMETERS as HP
from audio_datasets.aessd_dataset import Wav2Vec2Collator


def get_data_loaders(train_annotations_path, val_annotations_path,  feature_extractor):
    wav2vec_collator = Wav2Vec2Collator(feature_extractor)
    train_data = AESDD_Greek(
        BASE_DIR=PATHS.AESDD_DIR,
        ANNOTATIONS_PATH=os.path.join(PATHS.ROOT_DIR, train_annotations_path),
        feature_extractor=feature_extractor
    )
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=HP.BATCH_SIZE,
        shuffle=True,
        collate_fn=wav2vec_collator
    )

    print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
    return train_loader, train_loader # TODO: replace with val_loader after getting train test split CSVs