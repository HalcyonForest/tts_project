from final_block import Encoder, Decoder
from length_regulator import LengthRegulator

from final_block import get_mask_from_lengths

import torch.nn as nn
import torch

class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config, mel_config):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(model_config)
        self.length_regulator = LengthRegulator(model_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config.decoder_dim, mel_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0):
        encoded, mask = self.encoder(src_seq, src_pos)
        length, duration = self.length_regulator(encoded, alpha, length_target ,mel_max_length)
        decoded = self.decoder(length, duration)
        mel = self.mel_linear(decoded)
        return mel, duration