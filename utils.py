import wandb
import torch
import torch.nn as nn
import logging

def store_config(args, wandb):
    for key in vars(args).keys():
        wandb.config[key] = vars(args)[key]

def model_load(model, args):
    checkpoint = torch.load(args.model_dir)
    model.regressor.load_state_dict(checkpoint['regressor'])
    model.classifier.load_state_dict(checkpoint['classifier'])
    return model
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
