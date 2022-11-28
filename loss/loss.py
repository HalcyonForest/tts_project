import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, duration, pitch, energy, mel_target, duration_target, energy_target, pitch_target):
#         print("duration shapes: ", duration.shape, duration_target.shape)
#         print("pitch shapes: ", pitch.shape, pitch_target.shape)
#         print("energy shapes: ", energy.shape, energy_target.shape)
        mel_loss = self.mse_loss(mel, mel_target)

        duration_predictor_loss = self.l1_loss(duration,
                                               duration_target.float())
        pitch_predictor_loss = self.mse_loss(pitch, pitch_target)
        
        energy_predictor_loss = self.mse_loss(energy, energy_target)
        

        return mel_loss, duration_predictor_loss, energy_predictor_loss, pitch_predictor_loss
