import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from engine.solver import Trainer
from Data.build_dataloader import build_dataloader_cond
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one

class Args_Example:
    def __init__(self) -> None:
        self.gpu = 0
        self.config_path = './Config/wearable.yaml'
        self.save_dir = '/u/ybkim95/Diffusion-TS/OUTPUT/wearable'
        self.mode = 'infill'
        self.missing_ratio = 0.5
        self.milestone = 10
        self.name = 'wearable'
        os.makedirs(self.save_dir, exist_ok=True)

args = Args_Example()
configs = load_yaml_config(args.config_path)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

dl_info = build_dataloader_cond(configs, args)
model = instantiate_from_config(configs['model']).to(device)
trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)

trainer.load(args.milestone)

# Get data and settings
dataloader, dataset = dl_info['dataloader'], dl_info['dataset']
coef = configs['dataloader']['test_dataset']['coefficient']
stepsize = configs['dataloader']['test_dataset']['step_size']
sampling_steps = configs['dataloader']['test_dataset']['sampling_steps']
seq_length, feature_dim = dataset.window, dataset.var_num

# Define different demographic conditions to test
demographic_conditions = [
    {'age': 0, 'gender': 0, 'label': 'Young Male'},
    {'age': 0, 'gender': 1, 'label': 'Young Female'},
    {'age': 1, 'gender': 0, 'label': 'Older Male'},
    {'age': 1, 'gender': 1, 'label': 'Older Female'}
]

# Create directory for saving visualizations
viz_dir = os.path.join(args.save_dir, 'visualizations')
os.makedirs(viz_dir, exist_ok=True)

# Load original data and masks
ori_data = np.load(os.path.join(dataset.dir, f"wearable_norm_truth_{seq_length}_test.npy"))
masks = np.load(os.path.join(dataset.dir, f"wearable_masking_{seq_length}.npy"))
observed = ori_data * masks

# Feature names for better visualization
feature_names = ['Calories', 'Distance', 'BPM', 'Steps']

# Generate and visualize for each demographic condition
for condition in demographic_conditions:
    # Create context tensor for the condition
    context = torch.tensor([[condition['age'], condition['gender']]], device=device).float()
    
    # Generate samples with the condition
    samples, _, _ = trainer.restore(dataloader, [seq_length, feature_dim], coef, stepsize, sampling_steps, context=context)
    
    if dataset.auto_norm:
        samples = unnormalize_to_zero_to_one(samples)

    # Create visualization
    plt.rcParams["font.size"] = 12
    fig, axes = plt.subplots(nrows=feature_dim, ncols=1, figsize=(15, 20))
    fig.suptitle(f"Time Series Generation - {condition['label']}", fontsize=16)

    for feat_idx in range(feature_dim):
        # Plot observed points (masked)
        df_x = pd.DataFrame({
            "x": np.arange(0, seq_length), 
            "val": ori_data[0, :, feat_idx],
            "y": masks[0, :, feat_idx]
        })
        df_x = df_x[df_x.y!=0]

        # Plot missing points
        df_o = pd.DataFrame({
            "x": np.arange(0, seq_length), 
            "val": ori_data[0, :, feat_idx],
            "y": (1 - masks)[0, :, feat_idx]
        })
        df_o = df_o[df_o.y!=0]

        # Plotting
        axes[feat_idx].plot(df_o.x, df_o.val, color='blue', marker='o', 
                          linestyle='None', label='Missing Points')
        axes[feat_idx].plot(df_x.x, df_x.val, color='red', marker='x', 
                          linestyle='None', label='Observed Points')
        axes[feat_idx].plot(range(0, seq_length), samples[0, :, feat_idx], 
                          color='green', linestyle='solid', label='Diffusion-TS')
        
        axes[feat_idx].set_ylabel(f'{feature_names[feat_idx]} Value')
        axes[feat_idx].set_title(f'{feature_names[feat_idx]} Time Series')
        axes[feat_idx].legend()
        axes[feat_idx].grid(True, alpha=0.3)

    plt.xlabel('Time Steps')
    plt.tight_layout()
    
    # Save the figure
    fig_path = os.path.join(viz_dir, f'generation_{condition["label"].lower().replace(" ", "_")}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

# Create a comparison visualization
plt.figure(figsize=(20, 15))
feat_idx = 0  # Choose one feature to compare across demographics
for i, condition in enumerate(demographic_conditions):
    context = torch.tensor([[condition['age'], condition['gender']]], device=device).float()
    samples, _, _ = trainer.restore(dataloader, [seq_length, feature_dim], coef, stepsize, sampling_steps, context=context)
    
    if dataset.auto_norm:
        samples = unnormalize_to_zero_to_one(samples)
    
    plt.subplot(2, 2, i+1)
    plt.plot(range(0, seq_length), samples[0, :, feat_idx], 
             label=f'Generated ({condition["label"]})')
    plt.plot(df_x.x, df_x.val, 'rx', label='Observed Points')
    plt.title(f'{feature_names[feat_idx]} - {condition["label"]}')
    plt.xlabel('Time Steps')
    plt.ylabel(f'{feature_names[feat_idx]} Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'Tutorial1_new.png'), dpi=300, bbox_inches='tight')
plt.close()