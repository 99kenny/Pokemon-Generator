import wandb
import torch

def store_config(args, wandb):
    for key in vars(args).keys():
        wandb.config[key] = vars(args)[key]

def model_load(model, args):
    checkpoint = torch.load(args.model_dir)
    model.lora_layers.load_state_dict(checkpoint['lora'])
    model.regressor.load_state_dict(checkpoint['regressor'])
    model.classifier.load_state_dict(checkpoint['classifier'])
    return model
    
def norm(values, args):
    normed_values = []
    for value, (m, std) in zip(values, args.mean_std):
        normed_values.append((value - m) / std)

    return normed_values
    