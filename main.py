import sys 
import os
from parser_init import init_parser
from training import train

if __name__=='__main__':
    #init parser
    args = init_parser()

    if args.training :
        train(args)

    

