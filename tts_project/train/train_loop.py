from tqdm import tqdm 
from torch import nn
import torch
import os

def train(model, train_config, training_loader, fastspeech_loss, logger, optimizer, scheduler):

    tqdm_bar = tqdm(total=train_config.epochs * len(training_loader) * train_config.batch_expand_size - current_step)


    for epoch in range(train_config.epochs):
        for i, batchs in enumerate(training_loader):
            # real batch start here
            for j, db in enumerate(batchs):
                current_step += 1
                tqdm_bar.update(1)
                
                logger.set_step(current_step)

                # Get Data
                character = db["text"].long().to(train_config.device)
                mel_target = db["mel_target"].float().to(train_config.device)
                duration = db["duration"].int().to(train_config.device)
                mel_pos = db["mel_pos"].long().to(train_config.device)
                src_pos = db["src_pos"].long().to(train_config.device)
                max_mel_len = db["mel_max_len"]

                # Forward
                mel_output, duration_predictor_output = model(character,
                                                            src_pos,
                                                            mel_pos=mel_pos,
                                                            mel_max_length=max_mel_len,
                                                            length_target=duration)

                # Calc Loss
                mel_loss, duration_loss = fastspeech_loss(mel_output,
                                                        duration_predictor_output,
                                                        mel_target,
                                                        duration)
                total_loss = mel_loss + duration_loss

                # Logger
                t_l = total_loss.detach().cpu().numpy()
                m_l = mel_loss.detach().cpu().numpy()
                d_l = duration_loss.detach().cpu().numpy()

                logger.add_scalar("duration_loss", d_l)
                logger.add_scalar("mel_loss", m_l)
                logger.add_scalar("total_loss", t_l)

                # Backward
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.grad_clip_thresh)
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if current_step % train_config.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(train_config.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                    print("save model at step %d ..." % current_step)
