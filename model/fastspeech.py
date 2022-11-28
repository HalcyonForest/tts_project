from model.final_block import Encoder, Decoder
from model.length_regulator import LengthRegulator, EnergyPredictor, PitchPredictor

from model.final_block import get_mask_from_lengths

from audio import hparams_audio

import torch.nn as nn
import torch

# Pitch extraction lib
# import pywt
# import pyword
# Energy
import librosa

class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config, mel_config):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(model_config)
        
        self.length_regulator = LengthRegulator(model_config)
        self.energy_predictor = EnergyPredictor(model_config)
        self.pitch_predictor = PitchPredictor(model_config)
        
        
        self.energy_embedding = nn.Embedding(256, model_config.duration_predictor_filter_size)
        
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config.decoder_dim, mel_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)
    
    
    def get_pitch(x, model_config):
        F_0, time = pyworld.dio(x, model_config.sample_rate)
        return F_0 # Мейби (бейби)
        
        
    
    def get_energy(x, model_config):
        x_stft = librosa.stft(x, model_config.n_fft, model_config.hop_length, modek_config.win_length)
        x_stft = librosa.magphase(x_stft)[0]
        return torch.linalg.norm(x_stft, 2) # Ну вроде так
        
        
        
        

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0, pitch_target=None, energy_target=None):
#         print(1)
        encoded, _ = self.encoder(src_seq, src_pos)
        if self.training:
#             print("DPT shape: ", length_target.shape)
            length, duration = self.length_regulator(encoded, alpha, length_target ,mel_max_length)
            pitch, pitch_embedding = self.pitch_predictor(length, mel_pos, pitch_target)
            energy, energy_embedding = self.energy_predictor(length, mel_pos, energy_target)
            decoded = self.decoder(length + pitch_embedding + energy_embedding, mel_pos)
            mel = self.mel_linear(decoded)
            mel = self.mask_tensor(mel, mel_pos, mel_max_length)
        else:
#             print("Encoded: ", encoded)
            length, duration = self.length_regulator(encoded, alpha)
#             print("Length: ", length)
#             print("Fast stpeech forward: ", length, duration)
#             print("Going to decoder")
            pitch, pitch_embedding = self.pitch_predictor(length)
            energy, energy_embedding = self.energy_predictor(length)
#             print("Pitch:", pitch)
#             print("energy: ", energy)
#             print("mel pos: ", mel_pos)
            decoded = self.decoder(length + pitch_embedding + energy_embedding, duration)
            mel = self.mel_linear(decoded)

        return mel, duration, energy, pitch