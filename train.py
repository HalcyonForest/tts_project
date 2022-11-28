import torch 
from torch.optim.lr_scheduler  import OneCycleLR
from wandb_writer import WanDBWriter
from config.configs_classes import TrainConfig, MelSpectrogramConfig, FastSpeechConfig

from collator.collate import get_data_to_buffer, BufferDataset, collate_fn_tensor
from torch.utils.data import Dataset, DataLoader

from loss.loss import FastSpeechLoss
from model.fastspeech import FastSpeech
import os

from train_loop.train_function import train_loop
# train_config = TrainConfig()
mel_config = MelSpectrogramConfig()
model_config = FastSpeechConfig()
train_config = TrainConfig()

buffer = get_data_to_buffer(train_config)

dataset = BufferDataset(buffer)
one_batch_dataset = dataset[:256]
print(len(one_batch_dataset))
train_dataset = dataset[:int(len(dataset) * 0.8)]
val_dataset = dataset[int(len(dataset) * 0.8):]

training_loader = DataLoader(
    train_dataset,
    batch_size=train_config.batch_expand_size * train_config.batch_size,
    shuffle=True,
    collate_fn=collate_fn_tensor,
    drop_last=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=train_config.batch_expand_size * train_config.batch_size,
    shuffle=True,
    collate_fn=collate_fn_tensor,
    drop_last=True,
    num_workers=0
)

model = FastSpeech(model_config, mel_config)
model = model.to(train_config.device)

fastspeech_loss = FastSpeechLoss()
current_step = 0

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config.learning_rate,
    betas=(0.9, 0.98),
    eps=1e-9)
print(len(training_loader), train_config.batch_expand_size)
scheduler = OneCycleLR(optimizer, **{
    "steps_per_epoch": len(training_loader) * train_config.batch_expand_size,
    "epochs": train_config.epochs,
    "anneal_strategy": "cos",
    "max_lr": train_config.learning_rate,
    "pct_start": 0.1
})
logger = WanDBWriter(train_config)

train_loop(model, train_config, training_loader, val_loader, fastspeech_loss, logger, optimizer, scheduler)

