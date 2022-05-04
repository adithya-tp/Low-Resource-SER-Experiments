# Low-Resource-SER-Experiments
Project component for 11-785 (Introduction to Deep Learning) at CMU. Our experiments to build a better speech emotion recognition system for low resource languages. 


## Implementation of GE2E Loss:
The implementation is in the folder "GE2E". Pytorch implementation of Generalized End-to-End Loss for speaker verification, proposed in https://arxiv.org/pdf/1710.10467.pdf [3].
We referenced the code from https://github.com/cvqluu/GE2E-Loss

## Implementation of Wav2vec-Pretrained:
The implementation is experiments on low-resource datasets Italian(EMOVO) and Greek(AESDD) for emotion erification. This repository provides all the necessary tools to perform emotion recognition with a fine-tuned wav2vec2 (base) model using SpeechBrain. It is trained on IEMOCAP training data. It is referenced from https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP

## References
[1] https://github.com/Neclow/SERAB
[2] https://github.com/speechbrain/speechbrain
[3] GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION, https://arxiv.org/pdf/1710.10467.pdf