from text import text_to_sequence
import torch
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm 
import os
from torch.utils.data import Dataset
import librosa
import pyworld as pw
from librosa import stft
from audio import hparams_audio
import torchaudio

from config.configs_classes import TrainConfig
train_config = TrainConfig()

def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_1D_tensor(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_2D_tensor(inputs, maxlen=None):

    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def get_data_to_buffer(train_config):
    buffer = list()
    text = process_text(train_config.data_path)

    start = time.perf_counter()
    wav_names = os.listdir(train_config.wav_path)
    wav_names = sorted(wav_names)
#     print(wav_names)
#     return
    if not os.path.isdir("pitch"):
        os.makedirs("pitch")
            
    if not os.path.isdir("energy"):
        os.makedirs("energy")
        
    if not os.path.isdir("onepitch"):
        os.makedirs("onepitch")
            
    if not os.path.isdir("oneenergy"):
        os.makedirs("oneenergy")
        
    pitches = None
    energys = None
    
#     spectrogram = torchaudio.transforms.Spectrogram()

    for i in tqdm(range(len(text))):

        mel_gt_name = os.path.join(
            train_config.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(
            train_config.alignment_path, str(i)+".npy"))
        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(character, train_config.text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)
        

        
        if not os.path.isfile("./pitch/pitch"+str(i+1)+".npy") or not os.path.isfile("./energy/energy"+str(i+1)+".npy"):
            
            wav_gt_name = os.path.join(train_config.wav_path, wav_names[i])

            wav, sr = librosa.load(wav_gt_name)
            print("wav shape: ", wav.shape)
            energy = torch.stft(torch.tensor(wav), n_fft=1024, win_length=hparams_audio.win_length, hop_length=hparams_audio.hop_length).transpose(0,1)
            print("energy shape:", energy.shape)
            
            energy = torch.linalg.norm(energy, dim=-1)
            energy = torch.linalg.norm(energy, dim=-1)

            
            pitch, t = pw.dio(wav.astype(np.float64), sr, frame_period=hparams_audio.hop_length * 1000 / sr)
            pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sr).astype(np.float32)
            print("pitch shape: ", pitch.shape)
#             os.makedirs("pitch")
#             print(pitch[15])ы
            np.save("./pitch/pitch"+str(i+1)+".npy", pitch)
#             os.makedirs("energy")
            np.save("./energy/energy"+str(i+1)+".npy", energy)
        else:
            pitch = np.load("./pitch/pitch"+str(i+1)+".npy")
            energy = np.load("./energy/energy"+str(i+1)+".npy")
            
#             if pitches is None:
#                 pitches = np.array([pitch])
#             else:
#                 pitches = np.append(pitches, [pitch], axis=0)
                
            
#             if energys is None:
#                 energys = np.array([energy])
#             else:
#                 energys = np.append(energys, [energy], axis=0)
            
            
#             if i % 25 == 0:
#                 print("Energys: ", energys)
#                 print("Pitches: ", pitches)
                
        

        buffer.append({"text": character, "duration": duration,
                       "mel_target": mel_gt_target, "pitch":pitch, "energy":energy})
    
#     np.save("./onepitch/pitch.npy", pitches)
#     np.save("./oneenergy/energy.npy", energys)
    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]


def reprocess_tensor(batch, cut_list):
    energys = [batch[ind]["energy"] for ind in cut_list]
    pitches = [batch[ind]["pitch"] for ind in cut_list]
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    durations = [batch[ind]["duration"] for ind in cut_list]
    

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    mel_targets = pad_2D_tensor(mel_targets)
    pitch_targets = pad_1D(pitches)
    energys = pad_1D(energys)

    out = {"text": texts,
           "mel_target": mel_targets,
           "duration": durations,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_len": max_mel_len,
          "energy": energys,
          "pitch": pitch_targets}

    return out


def collate_fn_tensor(batch):
    len_arr = np.array([d["text"].size(0) for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = batchsize // train_config.batch_expand_size

    cut_list = list()
    for i in range(train_config.batch_expand_size):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(train_config.batch_expand_size):
        output.append(reprocess_tensor(batch, cut_list[i]))

    return output

