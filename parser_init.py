import argparse
import os
from datetime import datetime

epochs=100
lr=1e-04
optim_type='AdamW'
scheduler_type='get_linear_schedule_with_warmup'
BATCH_SIZE=16
MODEL_NAME="xyn-ai/anything-v4.0"
MODEL_DIR="output/models/"+'2023-11-27 09 34 06.pt'
MODEL_SAVE_DIR = "output/models/"+ "with_out_focal_loss"+ str(datetime.now())[:-7].replace(':', ' ') + '.pt'
HUB_MODEL_ID="pokemon-lora"
DATASET_DIR="dataset"
NUM_FEATURES = 6
NUM_CLASSES = 18

#INPUT_PROMPT = "'Electric-type', 'Pixelated', 'Code-covered fur', 'Exhausted expression', 'Coffee cup accessory', 'Keyboard tail', 'Bug-resistant', 'Analytical mindset', 'Scripting abilities', 'Debugging prowess'".replace("'","")
INPUT_PROMPT = "Plumber-type, Mushroom-shaped hat, Overalls attire, Moustache accessory, Jumping ability, Coin-collecting instinct, Pipe-traveler, Power-up transformations".replace("'","")
#INPUT_PROMPT = "'Ancient stone guardian', 'Massive mountain of strength', 'Rocky fortress with a heart of stone', 'Mysterious rock with hidden power', 'Unyielding protector of ancient ruins', 'Solemn figure sculpted from solid stone', 'Legendary golem of immense power', 'Eternal sentinel of the rocks', 'Embodiment of ancient earth energy', 'Invincible golem carved by time', 'Imposing presence amidst rocky terrain', 'Remnant of a forgotten era', 'Unbreakable guardian of secrets', 'Majestic stone colossus', 'Resolute in the face of adversity', 'Bearing the weight of time', 'Silent observer of eternal stillness', 'Symbol of resilience and endurance', 'Crystallized legend of untold tales', 'Unwavering pillar amid chaos'"
#INPUT_PROMPT = "stone, normal type"


def init_parser():
    parser = argparse.ArgumentParser(description='Parsing Method')

    parser.add_argument('--inference_prompt', default=INPUT_PROMPT.replace("'", ""))
    parser.add_argument('--inference_timestep', default=10, type=int)
    
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR)
    parser.add_argument('--model_save_dir', type=str, default=MODEL_SAVE_DIR)
    parser.add_argument('--output_dir', default='output/pokemon')
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR)
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--image_gen', type=bool, default=True)


    parser.add_argument('--pretrained_model_name_or_path', type=str, default=MODEL_NAME)
    parser.add_argument('--rank', default=4, type=int)
    parser.add_argument('--hidden_dim', default=768, type=int)
    parser.add_argument('--num_features', default = NUM_FEATURES, type= int)
    parser.add_argument('--num_classes' ,default=NUM_CLASSES, type=int)
    parser.add_argument("--revision",type=str,default=None,help="Revision of pretrained model identifier from huggingface.co/models.",)

    parser.add_argument('--cls_loss', default='CELoss', type = str)
    parser.add_argument('--optim_type', default=optim_type, type=str)
    parser.add_argument('--resolution', default=256, type=int)

    parser.add_argument('--epochs', default=epochs, type=int,
                            help='epochs Default is 100')
    parser.add_argument('--lr', default=lr, type=float,
                            help='epochs Default is 0.1')
    parser.add_argument('--batch_size', default=BATCH_SIZE, )
    parser.add_argument('--alpha_1', default=1, type=float)
    parser.add_argument('--alpha_2', default=1, type=float)

    #-------- Additional argument! Need to be refactored ---------#

    args = parser.parse_args()

    return args