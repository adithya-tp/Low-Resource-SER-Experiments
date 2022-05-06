"""BYOL for Audio: Training.

SYNOPSIS:
    train.py cfg.audio_dir MODEL_NAME <flags>

FLAGS:
    --config_path=CONFIG_PATH
        Default: 'config.yaml'
    --d=D
        Default: feature_d in the config.yaml
    --epochs=EPOCHS
        Default: epochs in the config.yaml
    --resume=RESUME
        Pathname to the weight file to continue training
        Default: Not specified

Example of training on FSD50K dataset:
    # Preprocess audio files to convert to 16kHz in advance.
    python -m utils.convert_wav /path/to/fsd50k work/16k/fsd50k
    # Run training on dev set for 300 epochs
    python train.py work/16k/fsd50k/FSD50K.dev_audio --epochs=300
"""
from enum import Enum
import multiprocessing
import os
import warnings
import pandas as pd

from pathlib import Path

import fire
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning
from torch.utils.data import DataLoader

from models.byol_a.augmentations import (MixupBYOLA, NormalizeBatch, RandomResizeCrop, RunningNorm, TimeFrequencyMasking)
from models.byol_a.byol_pytorch import BYOL
from models.byol_a.common import (get_logger, get_timestamp, load_yaml_config)
from models.byol_a.dataset import WaveInLMSOutDataset
from models.byol_a.models.audio_ntt import AudioNTT2020
from models.byol_a.models.cvt import CvT

class AugmentationModule:
    """BYOL-A augmentation module."""

    def __init__(self, size, epoch_samples, log_mixup_exp=True, mixup_ratio=0.4):
        self.train_transform = nn.Sequential(
            # TimeFrequencyMasking(freq_mask_param=48, time_mask_param=192),
            MixupBYOLA(ratio=mixup_ratio, log_mixup_exp=log_mixup_exp),
            RandomResizeCrop(virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5))
        )
        self.pre_norm = RunningNorm(epoch_samples=epoch_samples)

        self.__repr__()

    def __repr__(self):
        fns = [self.pre_norm.__class__.__name__] + [f.__class__.__name__ for f in self.train_transform]
        return (self.__class__.__name__ + '(\n{}\n)').format('\n'.join([f'  ({i}): {fn}' for i, fn in enumerate(fns)]))

    def __call__(self, x):
        x = self.pre_norm(x)
        return self.train_transform(x), self.train_transform(x)


class BYOLALearner(pl.LightningModule):
    """BYOL-A learner. Shows batch statistics for each epochs.

    Parameters
    ----------
    model: nn.Module
        BYOL training model

    lr: float
        Learning rate

    shape: tuple
        Image size represented as (height, width)

    **kwargs: dict, optional
        Extra arguments to BYOL

    Attributes
    ----------
    learner: nn.Module
        BYOL module, adapted from https://github.com/lucidrains/byol-pytorch/

    lr: float
        Learning rate

    post_norm: nn.Module
        Batch normalizer
    """

    def __init__(self, model, lr, shape, use_post_norm=True, **kwargs):
        super().__init__()
        self.learner = BYOL(model, image_size=shape, **kwargs)
        self.lr = lr
        if use_post_norm:
            self.post_norm = NormalizeBatch()
        else:
            self.post_norm = None

    def forward(self, images1, images2):
        """Forward pass.

        Parameters
        ----------
        images1: torch.Tensor
            First augmented image

        images2: torch.Tensor
            Second augmented image

        Returns
        ----------
        torch.Tensor
            Forward pass output of the BYOL module
        """
        return self.learner(images1, images2)

    def training_step(self, paired_inputs, batch_idx):
        """Lightning training step.

        Parameters
        ----------
        paired_inputs: list[torch.Tensor]
            Pair of image inputs

        batch_idx: int
            Batch index

        Returns
        ----------
        loss: torch.tensor
            Model loss for a given batch at a given epoch
        """
        def to_np(A):
            return [a.cpu().numpy() for a in A]

        bs = paired_inputs[0].shape[0]
        paired_inputs = torch.cat(paired_inputs)  # [(B,1,T,F), (B,1,T,F)] -> (2*B,1,T,F)
        mb, sb = to_np((paired_inputs.mean(), paired_inputs.std()))
        if self.post_norm:
            paired_inputs = self.post_norm(paired_inputs)
        ma, sa = to_np((paired_inputs.mean(), paired_inputs.std()))

        loss = self.forward(paired_inputs[:bs], paired_inputs[bs:])
        for k, v in {'mb': mb, 'sb': sb, 'ma': ma, 'sa': sa}.items():
            self.log(k, float(v), prog_bar=True, on_step=False, on_epoch=True)
        self.log('loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Lightning optimizer configuration.

        Returns
        ----------
        torch.optim object
            Optimizer based on model parameters and learning rate
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_before_zero_grad(self, _):
        """Called after ``training_step()`` and before ``optimizer.zero_grad()``.

        Updates moving average function from the BYOL learner before zeroing grads.
        """
        self.learner.update_moving_average()


class LANG(Enum):
    ENGLISH = 0
    FRENCH = 1
    GERMAN = 2
    GREEK = 3
    ITALIAN = 4

def get_english_path(base_name, LANG_DATA_DIR):
    base_split = base_name.split('_')
    session_num = base_split[0][-2]
    session = f'Session{session_num}'
    sub_folder = "_".join(base_split[:-1])
    path = os.path.join(LANG_DATA_DIR, f'{session}/sentences/wav/{sub_folder}/{base_name}.wav')
    return path

def get_modified_greek_df(df, LANG_DATA_DIR):
    file_paths = []
    for idx, row in df.iterrows():
        path_name, label = row
        file_paths.append(os.path.join(LANG_DATA_DIR, f'{label}/{path_name}'))
    
    d = {'file': file_paths, 'label': df['folder_or_label'].tolist()}
    df = pd.DataFrame(d)
    return df

def get_modified_italian_df(df, LANG_DATA_DIR):
    file_paths = []
    for idx, row in df.iterrows():
        path_name, folder, label = row
        file_paths.append(os.path.join(LANG_DATA_DIR, folder, path_name))
    
    d = {'file': file_paths, 'label': df['emotion'].tolist()}
    df = pd.DataFrame(d)
    return df

def main(model_name, config_path='models/byol_a/config.yml', d=None, epochs=None, resume=None):
    warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
    cfg = load_yaml_config(config_path)
    # Override configs
    num_feature_d = d or cfg.feature_d
    cfg.epochs = epochs or cfg.epochs
    cfg.resume = resume or cfg.resume
    # Essentials
    logger = get_logger(__name__)
    logger.info(cfg)
    seed_everything(cfg.seed)
    # Data preparation

    # Run train_utils script from main directory
    # lang = getattr(LANG, lang)
    lang = LANG.ITALIAN

    CURR_DIR = os.getcwd()
    if lang == LANG.ENGLISH:
        emotion_dict = {
            'sad': "Sadness", 'hap': "Happiness", 'fru': "Frustration", 
            'exc': "Excitement", 'ang': "Anger", 'xxx': "Unknown", 
            'neu': "Neutral", 'sur': "Surprise", 'oth': "Other",  
            'fea': "Fear", 'dis': "Disgust"
        }
        df = pd.read_csv(os.path.join(CURR_DIR, 'annotations/english_annotations.csv'))
        df["label"] = df["emotion"].map(emotion_dict)
        df.rename(columns = {'wav_file':'file'}, inplace = True)
        del df["emotion"]
        LANG_DATA_DIR = os.path.join(os.getcwd(), 'data/english/IEMOCAP_full_release')
        df['file'] = df['file'].apply(lambda x: get_english_path(x, LANG_DATA_DIR))

    elif lang == LANG.GERMAN:
        emotion_dict = {
            'afraid': "Fear", 'angry': "Anger", 'boring': "Bore", 
            'disgust': "Disgust", 'happy': "Happiness", 
            'neutral': "Neutral", 'sad': "Sadness"
        }
        df = pd.read_csv(os.path.join(CURR_DIR, 'annotations/german_annotations.csv'))
        df["label"] = df["emotion_en"].map(emotion_dict)
        del df["emotion_en"]
        df.rename(columns = {'uuid':'file'}, inplace = True)
        
        LANG_DATA_DIR = os.path.join(os.getcwd(), 'data/german/wav/')
        df['file'] = df['file'].map(lambda x: os.path.join(LANG_DATA_DIR, "{}.wav".format(x)))

    elif lang == LANG.FRENCH:
        df = pd.read_csv(os.path.join(CURR_DIR, 'annotations/french_annotations.csv'))
        LANG_DATA_DIR = os.path.join(os.getcwd(), 'data/french')
        df['file'] = df['file'].map(lambda x: os.path.join(LANG_DATA_DIR, x))

    elif lang == LANG.GREEK:
        df = pd.read_csv(os.path.join(CURR_DIR, 'annotations/annotations_aesdd.csv'))
        df.rename(columns = {'file_path':'file'}, inplace = True)
        LANG_DATA_DIR = os.path.join(os.getcwd(), 'data/greek')
        df = get_modified_greek_df(df, LANG_DATA_DIR)
    
    elif lang == LANG.ITALIAN:
        df = pd.read_csv(os.path.join(CURR_DIR, 'annotations/annotations_emovo.csv'))
        df.rename(columns = {'file_path':'file'}, inplace = True)
        LANG_DATA_DIR = os.path.join(os.getcwd(), 'data/italian')
        df = get_modified_italian_df(df, LANG_DATA_DIR)


    else:
        raise ValueError('Please pass in a valid language (ENGLISH, GERMAN or FRENCH)')

    
    files = df['file'].tolist()
    # Sanity check
    assert(all([os.path.exists(_) for _ in files[:50]]))

    transform = AugmentationModule(cfg.shape, 2 * len(files))
    ds = WaveInLMSOutDataset(cfg, files, labels=None, transform=transform)  # , use_librosa=True)
    dl = DataLoader(
        ds,
        batch_size=cfg.bs,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=False,
        shuffle=True,
    )
    logger.info(f'Dataset: {len(files)} .wav files from {LANG_DATA_DIR}')

    # Load model
    if model_name == 'default':
        model = AudioNTT2020(n_mels=cfg.n_mels, d=cfg.feature_d)

        model_name += str(cfg.feature_d)

    elif model_name == 'cvt':
        s1_depth, s2_depth, s3_depth = cfg.depths
        s1_emb_dim, s2_emb_dim, s3_emb_dim = cfg.embed_dims
        s1_mlp_mult, s2_mlp_mult, s3_mlp_mult = cfg.mlp_mults

        model = CvT(
            s1_emb_dim=s1_emb_dim,
            s1_depth=s1_depth,
            s1_mlp_mult=s1_mlp_mult,
            s2_emb_dim=s2_emb_dim,
            s2_depth=s2_depth,
            s2_mlp_mult=s2_mlp_mult,
            s3_emb_dim=s3_emb_dim,
            s3_depth=s3_depth,
            s3_mlp_mult=s3_mlp_mult,
            pool=cfg.cvt_pool
        )

        model_name += f'_s1-d{s1_depth}-e{s1_emb_dim}_s2-d{s2_depth}-e{s2_emb_dim}_s3-d{s3_depth}-e{s3_emb_dim}'
    else:
        raise ValueError('Model not found.')

    # Training preparation
    name = (f'{model_name}_BYOLAs{cfg.shape[0]}x{cfg.shape[1]}-{get_timestamp()}'
            f'-e{cfg.epochs}-bs{cfg.bs}-lr{str(cfg.lr)[2:]}'
            f'-rs{cfg.seed}')

    transform_names = [f.__class__.__name__ for f in transform.train_transform]

    if transform_names != ['MixupBYOLA', 'RandomResizeCrop']:
        name += '-aug' + '_'.join(map(str.lower, transform_names))

    logger.info(f'Training {name}...')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg.resume is not None:
        model.load_weight(cfg.resume, device)

    # Training
    learner = BYOLALearner(
        model,
        cfg.lr,
        cfg.shape,
        use_post_norm=cfg.use_post_norm,
        hidden_layer=-1,
        projection_size=cfg.proj_size,
        projection_hidden_size=cfg.proj_dim,
        moving_average_decay=cfg.ema_decay,
    )
    trainer = pl.Trainer(
        gpus=cfg.gpus,
        # accelerator="ddp_spawn",
        max_epochs=cfg.epochs,
        weights_summary=None,
    )
    trainer.fit(learner, dl)
    if trainer.interrupted:
        logger.info('Terminated.')
        exit(0)
    # Saving trained weight.
    to_file = Path(cfg.checkpoint_folder) / (name + '.pth')
    to_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), to_file)
    logger.info(f'Saved weight as {to_file}')


if __name__ == '__main__':
    fire.Fire(main)