import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from engine.solver import Trainer
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset, DataLoader
from Data.build_dataloader import build_dataloader_cond
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one

class WearableDataset(Dataset):
    def __init__(self, data, context, regular=True, pred_length=24):
        super(WearableDataset, self).__init__()
        self.sample_num = data.shape[0]
        self.samples = data
        self.context = context
        self.regular = regular
        self.mask = np.ones_like(data)
        self.mask[:, -pred_length:, :] = 0.
        self.mask = self.mask.astype(bool)

    def __getitem__(self, ind):
        x = self.samples[ind, :, :]
        c = self.context[ind, :]
        if self.regular:
            return torch.from_numpy(x).float(), torch.from_numpy(c).float()
        mask = self.mask[ind, :, :]
        return torch.from_numpy(x).float(), torch.from_numpy(c).float(), torch.from_numpy(mask)

    def __len__(self):
        return self.sample_num

###

class Args_Example:
    def __init__(self) -> None:
        self.gpu = 0
        self.config_path = './Config/wearable.yaml'
        self.save_dir = '/u/ybkim95/Diffusion-TS/OUTPUT/wearable'
        self.mode = 'predict'
        self.pred_len = 24  # Predict next 24 hours
        self.milestone = 10
        self.name = 'wearable'
        os.makedirs(self.save_dir, exist_ok=True)

args = Args_Example()
configs = load_yaml_config(args.config_path)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

# Load your original dataset with context
dl_info = build_dataloader_cond(configs, args)
dataset = dl_info['dataset']

# Get the time series and context data
ts_data = dataset.samples[0]  # time series data
context_data = dataset.samples[1]  # context data

# Create train/test splits
train_size = int(0.8 * len(ts_data))
train_ts = ts_data[:train_size]
test_ts = ts_data[train_size:]
train_context = context_data[:train_size]
test_context = context_data[train_size:]

# Create datasets
train_dataset = WearableDataset(train_ts, train_context, regular=True)
test_dataset = WearableDataset(test_ts, test_context, regular=False, pred_length=args.pred_len)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                            num_workers=0, drop_last=True, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_ts.shape[0], 
                           shuffle=False, num_workers=0, pin_memory=True)

# Initialize model and trainer
model = instantiate_from_config(configs['model']).to(device)
trainer = Trainer(config=configs, args=args, model=model, 
                 dataloader={'dataloader': train_dataloader})

# Train model
trainer.train()

# Generate predictions
seq_length, feat_num = test_ts.shape[1], test_ts.shape[2]
samples, reals, masks = trainer.restore(test_dataloader, 
                                      shape=[seq_length, feat_num],
                                      coef=1e-2, 
                                      stepsize=5e-2, 
                                      sampling_steps=200)

if dataset.auto_norm:
    samples = unnormalize_to_zero_to_one(samples)
    reals = unnormalize_to_zero_to_one(reals)

# Calculate MSE
mse = mean_squared_error(samples[~masks], reals[~masks])
print(f"MSE: {mse}")

# Visualization
plt.rcParams["font.size"] = 12
feature_names = ['Calories', 'Distance', 'BPM', 'Steps']

for feat_idx, feat_name in enumerate(feature_names):
    plt.figure(figsize=(15, 3))
    
    # Plot history
    plt.plot(range(0, seq_length-args.pred_len), 
            reals[0, :(seq_length-args.pred_len), feat_idx], 
            color='c', linestyle='solid', label='History')
    
    # Plot ground truth
    plt.plot(range(seq_length-args.pred_len-1, seq_length),
            reals[0, -args.pred_len-1:, feat_idx], 
            color='g', linestyle='solid', label='Ground Truth')
    
    # Plot predictions
    plt.plot(range(seq_length-args.pred_len-1, seq_length),
            samples[0, -args.pred_len-1:, feat_idx], 
            color='r', linestyle='solid', label='Prediction')
    
    plt.title(f'{feat_name} Forecasting')
    plt.xlabel('Time (hours)')
    plt.ylabel(feat_name)
    plt.tick_params('both', labelsize=15)
    plt.subplots_adjust(bottom=0.1, left=0.05, right=0.99, top=0.95)
    plt.legend()
    plt.savefig(f'wearable_forecast_{feat_name.lower()}.png')
    plt.close()