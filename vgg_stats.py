import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd
from collections import OrderedDict
import csv

# Function to count sparse elements in a tensor with a threshold
def count_sparse_elements(tensor, threshold=1e-6):
    return torch.sum(torch.abs(tensor) < threshold).item()

# Function to calculate weight sparsity with threshold
def calculate_sparsity(tensor, threshold=1e-6):
    total_elements = tensor.numel()
    sparse_elements = count_sparse_elements(tensor, threshold)
    return sparse_elements / total_elements

# Function to register hooks for activation sparsity
def register_activation_hooks(model):
    activation_stats = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            # Store activations for analysis
            if isinstance(output, torch.Tensor):
                activation_stats[name] = output.detach().clone()
            else:
                # Handle tuple outputs
                activation_stats[name] = output[0].detach().clone() if isinstance(output, tuple) else None
        return hook
    
    # Register hooks for all modules to capture activations
    for name, module in model.named_modules():
        if not list(module.children()): # Only register for leaf modules
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    return activation_stats, hooks

# Function to calculate MACs (multiply-accumulate operations)
def calculate_macs(module, input_shape, output_shape):
    if isinstance(module, nn.Conv2d):
        batch_size, in_channels, in_h, in_w = input_shape
        batch_size, out_channels, out_h, out_w = output_shape
        kernel_h, kernel_w = module.kernel_size
        # Each output element requires kernel_h * kernel_w * in_channels MACs
        return out_h * out_w * out_channels * kernel_h * kernel_w * in_channels
    
    elif isinstance(module, nn.Linear):
        return module.in_features * module.out_features
    
    return 0

# Function to get model stats
def get_model_stats(model):
    # Create a dummy input with values between 0-1 (more realistic than random noise)
    dummy_input = torch.rand(1, 3, 224, 224)
    
    # Register hooks for activation sparsity
    activation_stats, hooks = register_activation_hooks(model)
    
    # Forward pass to trigger hooks
    with torch.no_grad():
        output = model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    stats = []
    
    # Input shapes for MAC calculation
    input_shapes = {}
    output_shapes = {}
    
    # Register hooks to capture input and output shapes
    def register_shape_hooks(model):
        shape_hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(input[0], torch.Tensor):
                    input_shapes[name] = input[0].shape
                    output_shapes[name] = output.shape
            return hook
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                shape_hooks.append(module.register_forward_hook(hook_fn(name)))
        
        return shape_hooks
    
    # Register and run shape hooks
    shape_hooks = register_shape_hooks(model)
    with torch.no_grad():
        model(dummy_input)
    for hook in shape_hooks:
        hook.remove()
    
    # Collect stats for each layer
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Weight stats
            weight_params = module.weight.numel()
            weight_sparsity = calculate_sparsity(module.weight, threshold=1e-4)
            
            # MACs calculation
            macs = 0
            if name in input_shapes and name in output_shapes:
                macs = calculate_macs(module, input_shapes[name], output_shapes[name])
            
            # Activation sparsity - look for ReLU outputs which should have true zeros
            act_sparsity = 0
            
            # For each layer, look for corresponding activation
            # Usually the activations after ReLU will show sparsity
            if name in activation_stats:
                activation = activation_stats[name]
                if activation is not None:
                    act_sparsity = calculate_sparsity(activation)
            
            # Special handling for ReLU activations which should show true sparsity
            relu_name = name.replace('conv', 'relu').replace('linear', 'relu')
            if relu_name in activation_stats:
                activation = activation_stats[relu_name]
                if activation is not None:
                    act_sparsity = calculate_sparsity(activation)
            
            stats.append({
                'Layer': name,
                'Weights': weight_params,
                'MACs': macs,
                'Weight_Sparsity': weight_sparsity,
                'Activation_Sparsity': act_sparsity
            })
    
    return stats

# Load VGG16 with batch norm (since VGG18 is not standard in PyTorch)
print("Loading VGG model...")
model = models.vgg16_bn(weights='DEFAULT')  # Using newer weights parameter instead of deprecated 'pretrained'
model.eval()  # Set to evaluation mode

# Add a threshold-based pruning to artificially create some weight sparsity
# This will help demonstrate the sparsity calculation
print("Applying threshold-based pruning to create sparse weights...")
threshold = 1e-3  # Small threshold for pruning

# Apply pruning to each parameter
with torch.no_grad():
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Create a mask for small weights
            mask = torch.abs(param.data) < threshold
            # Zero out weights below threshold
            param.data[mask] = 0.0

# Get model statistics
print("Calculating model statistics...")
stats = get_model_stats(model)

# Create DataFrame 
df = pd.DataFrame(stats)

# Add additional analysis - summary statistics
total_weights = df['Weights'].sum()
total_macs = df['MACs'].sum()
avg_weight_sparsity = (df['Weight_Sparsity'] * df['Weights']).sum() / total_weights
avg_act_sparsity = df['Activation_Sparsity'].mean()  # Simple mean for activation sparsity

# Format percentages for better readability
df['Weight_Sparsity'] = df['Weight_Sparsity']
df['Activation_Sparsity'] = df['Activation_Sparsity']

# Rename columns to include '%' in the header
df.rename(columns={'Weight_Sparsity': 'Weight_Sparsity(0-1)', 'Activation_Sparsity': 'Activation_Sparsity(0-1)'}, inplace=True)

# Save to CSV
df.to_csv('vgg_stats.csv', index=False)


print("Analysis complete. Results saved to 'vgg_stats.csv'")
print(f"Total weights: {total_weights:,}")
print(f"Total MACs: {total_macs:,}")
print(f"Average weight sparsity: {avg_weight_sparsity*100:.4f}%")
print(f"Average activation sparsity: {avg_act_sparsity*100:.4f}%")

# Display the results
print("\nLayer statistics:")
print(df)