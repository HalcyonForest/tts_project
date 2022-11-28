import torch
from torch import nn
import torch.nn.functional as F
from audio import hparams_audio

def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat

class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, model_config):
        super(DurationPredictor, self).__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
            
        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out
        
        
class EnergyPredictor(nn.Module):
    
    def __init__(self, model_config):
        super(EnergyPredictor, self).__init__()
        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout
        
        self.energy_linspace = nn.Parameter(torch.linspace(0, 315, 255))
        
        self.energy_embedding = nn.Embedding(256, model_config.duration_predictor_filter_size)
        
        self.predictor = DurationPredictor(model_config)
        
        
    def forward(self, x, mask=None, energy_target=None):
        x = self.predictor(x).squeeze(-1)
        if mask is not None:
            x = x.masked_fill(mask.bool(), 0.)
            
        if energy_target is not None:
            energy_embedding = self.energy_embedding(torch.bucketize(energy_target.detach(), self.energy_linspace.detach()))
        else:
            energy_embedding = self.energy_embedding(torch.bucketize(torch.exp(x.detach())-1, self.energy_linspace.detach()))
        return x, energy_embedding
    
class PitchPredictor(nn.Module):
    
    def __init__(self, model_config):
        super(PitchPredictor, self).__init__()
        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout
        
        self.pitch_embedding = nn.Embedding(256, model_config.duration_predictor_filter_size)
        
        self.pitch_linspace = nn.Parameter(torch.linspace(hparams_audio.mel_fmin, hparams_audio.mel_fmax , 255))
        
        self.predictor = DurationPredictor(model_config)
        
        
    def forward(self, x, mask=None, pitch_target=None):
        print("X shape: ", x.shape)
        x = self.predictor(x).squeeze(-1)
        if mask is not None:
            x = x.masked_fill(mask.bool(), 0.)
        
        if pitch_target is not None:
            pitch_embedding = self.pitch_embedding(torch.bucketize(pitch_target.detach(), self.pitch_linspace.detach()))
        else:
            pitch_embedding = self.pitch_embedding(torch.bucketize(torch.exp(x.detach()) - 1, self.pitch_linspace.detach()))
        return x, pitch_embedding
        
        
        


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, model_config):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = DurationPredictor(model_config)
        self.model_config = model_config

    def LR(self, x, duration_predictor_output, mel_max_length=None):
#         print("DPO shape: ", duration_predictor_output.shape)
        expand_max_len = torch.max(torch.sum(duration_predictor_output, -1), -1)[0]
        
        alignment = torch.zeros(duration_predictor_output.size(0), expand_max_len, duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment, duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
#         print("Enetred LR forward")
        duration_predictor_output = self.duration_predictor(x)
        
#         print("DPO: ", duration_predictor_output)

        if target is not None:
            output = self.LR(x, target, mel_max_length=mel_max_length)
            return output, duration_predictor_output
        else:
#             duration_predictor_output = torch.exp(duration_predictor_output) - 1
            duration_predictor_output = (
                (duration_predictor_output + 0.5) * alpha).int()
            output = self.LR(x, duration_predictor_output)
#             print("Output: ", output)
            mel_pos = torch.stack(
                [torch.Tensor([i+1 for i in range(output.size(1))])]).long().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            return output, mel_pos
        