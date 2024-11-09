import os
import sys
import time
import torch
import wandb
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data

def compute_cosine_similarity(pred, target):
    """
    Compute cosine similarity between predictions and targets.
    Returns similarity between -1 and 1, with values closer to 1 indicating higher similarity.
    """
    # Reshape if needed - ensure we're computing along the feature dimension
    # Assuming shape is (batch_size, seq_length, feature_size)
    if len(pred.shape) == 3:
        # Reshape to (batch_size * seq_length, feature_size)
        pred = pred.reshape(-1, pred.shape[-1])
        target = target.reshape(-1, target.shape[-1])

    # Normalize vectors
    pred_normalized = F.normalize(pred, p=2, dim=-1)
    target_normalized = F.normalize(target, p=2, dim=-1)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(pred_normalized, target_normalized, dim=-1)
    
    # Return mean similarity (should be between -1 and 1)
    return similarity.mean()

class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader'])
        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger

        # Initialize wandb
        wandb.init(
            project="Diffusion-TS",
            name=args.name,
            config={
                'model_type': model.__class__.__name__,
                'seq_length': model.seq_length,
                'learning_rate': config['solver'].get('base_lr', 1.0e-4),
                'max_epochs': self.train_num_steps,
                'gradient_accumulation': self.gradient_accumulate_every,
                'context_conditioning': model.context_dims > 0
            }
        )

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    def train(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                total_cosine_sim = 0.
                
                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.dl)
                    
                    # Handle different return types from dataset
                    if isinstance(batch, (list, tuple)):
                        if len(batch) == 2:  # Training data with context
                            data, context = batch
                            data, context = data.to(device), context.to(device)
                        elif len(batch) == 3:  # Test data with context and mask
                            data, context, mask = batch
                            data, context, mask = data.to(device), context.to(device), mask.to(device)
                    else:
                        data = batch.to(device)
                        context = None
                    
                    # Forward pass through model with context
                    loss = self.model(data, context=context, target=data)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    
                    # Get noise prediction
                    with torch.no_grad():
                        t = torch.randint(0, self.model.num_timesteps, (data.shape[0],), device=device).long()
                        noise = torch.randn_like(data)
                        x_t = self.model.q_sample(data, t, noise=noise)
                        pred_noise, x_start = self.model.model_predictions(x_t, t, context=context)
                        
                        cosine_sim = compute_cosine_similarity(pred_noise, noise)
                        total_cosine_sim += cosine_sim.item()
                    
                    total_loss += loss.item()

                avg_loss = total_loss / self.gradient_accumulate_every
                avg_cosine_sim = total_cosine_sim / self.gradient_accumulate_every
                
                pbar.set_description(f'loss: {avg_loss:.6f}, noise_cosine_sim: {avg_cosine_sim:.4f}')

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(avg_loss)
                self.opt.zero_grad()

                # Log metrics
                wandb.log({
                    'loss': avg_loss,
                    'noise_cosine_similarity': avg_cosine_sim,
                    'learning_rate': self.opt.param_groups[0]['lr'],
                }, step=self.step)

                if self.logger is not None:
                    self.logger.add_scalar(tag='train/loss', scalar_value=avg_loss, global_step=self.step)
                    self.logger.add_scalar(tag='train/noise_cosine_similarity', scalar_value=avg_cosine_sim, global_step=self.step)

                self.step += 1
                step += 1
                self.ema.update()

                if self.step != 0 and self.step % self.save_cycle == 0:
                    self.milestone += 1
                    self.save(self.milestone)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))
        
        wandb.finish()

    def sample(self, num, size_every, shape=None, context=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):
            # Pass context to generate_mts
            sample = self.ema.ema_model.generate_mts(
                batch_size=size_every, 
                context=context.to(self.device) if context is not None else None
            )
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        return samples

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        for batch in raw_dataloader:
            if len(batch) == 3:  # data with context and mask
                x, context, t_m = batch
                x, context, t_m = x.to(self.device), context.to(self.device), t_m.to(self.device)
            else:  # data with mask only
                x, t_m = batch
                x, t_m = x.to(self.device), t_m.to(self.device)
                context = None

            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(
                    shape=x.shape, 
                    target=x*t_m, 
                    partial_mask=t_m,
                    # context=context,
                    model_kwargs=model_kwargs
                )
            else:
                sample = self.ema.ema_model.fast_sample_infill(
                    shape=x.shape, 
                    target=x*t_m, 
                    partial_mask=t_m, 
                    # context=context,
                    model_kwargs=model_kwargs,
                    sampling_timesteps=sampling_steps
                )

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])
        
        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks
