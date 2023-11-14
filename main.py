import sys 
import os
from parser_init import init_parser
from training import train
from testing import test
from utils import store_config
import wandb
from datetime import datetime


now = datetime.now()

if __name__=='__main__':
    #init parser
    args = init_parser()
    
    if args.wandb :
        wandb.init(project='pokemon')
        wandb.run.name = str(now)
        store_config(args, wandb)
        
    if args.training :
        train(args)
    
    if args.testing :
        test(args)

    

