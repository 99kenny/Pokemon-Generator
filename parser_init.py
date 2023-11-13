import argparse

epochs=4
lr=1e-04
optim_type='AdamW'
scheduler_type='get_linear_schedule_with_warmup'
BATCH_SIZE=16
MODEL_NAME="runwayml/stable-diffusion-v1-5"
OUTPUT_DIR="/output"
HUB_MODEL_ID="pokemon-lora"
DATASET_DIR="dataset"
NUM_FEATURES = 6
NUM_CLASSES = 18
def init_parser():
    parser = argparse.ArgumentParser(description='Parsing Method')
    
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR)

    parser.add_argument('--pretrained_model_name_or_path', type=str, default=MODEL_NAME)
    parser.add_argument('--rank', default=4, type=int)
    parser.add_argument('--latent_dim', default=4096, type=int)
    parser.add_argument('--num_features', default = NUM_FEATURES, type= int)
    parser.add_argument('--num_classes' ,default=NUM_CLASSES, type=int)
    parser.add_argument("--revision",type=str,default=None,help="Revision of pretrained model identifier from huggingface.co/models.",)


    parser.add_argument('--optim_type', default=optim_type, type=str)
    parser.add_argument('--resolution', default=256, type=int)

    parser.add_argument('--epochs', default=epochs, type=int,
                            help='epochs Default is 100')
    parser.add_argument('--lr', default=lr, type=float,
                            help='epochs Default is 0.1')
    parser.add_argument('--batch_size', default=BATCH_SIZE, )

    #-------- Additional argument! Need to be refactored ---------#

    args = parser.parse_args()

    return args