import torch 
from torch.optim.lr_scheduler  import OneCycleLR
from wandb_writer import WanDBWriter
from config.configs_classes import TrainConfig

from collator.collate import get_data_to_buffer, BufferDataset, collate_fn_tensor
from torch.utils.data import Dataset, DataLoader

from loss.loss import FastSpeechLoss
from model.fastspeech import FastSpeech



train_config = TrainConfig()

buffer = get_data_to_buffer(train_config)

dataset = BufferDataset(buffer)

training_loader = DataLoader(
    dataset,
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

scheduler = OneCycleLR(optimizer, **{
    "steps_per_epoch": len(training_loader) * train_config.batch_expand_size,
    "epochs": train_config.epochs,
    "anneal_strategy": "cos",
    "max_lr": train_config.learning_rate,
    "pct_start": 0.1
})

train_loop(...)

logger = WanDBWriter(train_config)