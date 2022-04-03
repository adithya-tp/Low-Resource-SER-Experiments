from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PreTrainedModel, Wav2Vec2Model
import torch
import torch.nn as nn

# Wav2vec model skeleton
class Wav2vec2SER(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.feat_extractor = Wav2Vec2Model(config)
        
        self.cls_layer = nn.Sequential(*[
            nn.Dropout(config.final_dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.final_dropout),
            nn.Linear(config.hidden_size, config.num_labels)
        ]) 
        self.init_weights()

    def forward(self, x):
        backbone_out = self.feat_extractor(x)
        out = self.cls_layer(torch.mean(backbone_out[0], dim=1))
        logits = self.cls_layer(out)
        return logits