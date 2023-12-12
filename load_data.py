import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from utils import norm
from sklearn.preprocessing import RobustScaler

ptype_dict_str_int = {'bug': 0,
    'dark': 1,
    'dragon': 2,
    'electric': 3,
    'fairy': 4,
    'fighting': 5,
    'fire': 6,
    'flying': 7,
    'ghost': 8,
    'grass': 9,
    'ground': 10,
    'ice': 11,
    'normal': 12,
    'poison': 13,
    'psychic': 14,
    'rock': 15,
    'steel': 16,
    'water': 17}

def category(name):
    return ptype_dict_str_int[name]

class PokemonDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, args, root_dir, prompt_num=20, transform=None, Tokenizer = None):
        self.prompt_num = prompt_num
        self.root_dir = root_dir
        self.transform = transform
        self.Tokenizer = Tokenizer
        self.args = args

        self.tabular = pd.read_csv(f'{root_dir}/pokemon_preprocessed.csv').sort_values(by='name').reset_index(drop=True)
        self.p_type = self.tabular['type1'].apply(category)
        self.names = self.tabular['name']
        self.tabular = self.tabular.drop(['type1','name'], axis=1)
        self.scaler = RobustScaler().fit(self.tabular)
        self.tabular = self.scaler.transform(self.tabular)
        self.prompts = pd.read_csv(f'{root_dir}/pokemon_prompts.csv').sort_values(by='name').reset_index(drop=True)
        
        assert len(self.names.compare(self.prompts['name'])) == 0

        self.prompts = self.prompts['prompts'].str.replace("'","").str.replace('[','').str.replace(']','')

        

        image_dir = os.path.join(root_dir, 'images')
        self.images = []
        for folder in self.names:
            file_path = os.path.join(image_dir,folder)
            if len(file_path) == 0:
                raise ValueError
            self.images.append(file_path) 
        
        
        
    def tokenize_captions(self, captions,is_train=True):
        inputs = self.Tokenizer('<|startoftext|>, high resolution, masterpiece, best quality, ' + captions, max_length=self.Tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids
    
    def __len__(self):
        return len(self.tabular)

    def __getitem__(self, idx):
        tabular = self.tabular[idx]
        p_type = self.p_type.iloc[idx]
        prompt = self.prompts.iloc[idx]
        
        
        output = dict()
        output['tabular'] = torch.Tensor(tabular) #  get rid of name of pokemon
        output['prompt'] = ','.join(random.sample(prompt.split(','), self.prompt_num))
        output['prompt'] = self.tokenize_captions(output['prompt'])
        output['p_type'] = int(p_type)
       
        
        # random image in image dir
        image = self.images[idx]
        file_list = os.listdir(image)
        image_idx = random.randint(0,len(file_list)-1)
        file_path = os.path.join(image,file_list[image_idx])
        img = Image.open(file_path).convert('RGB')
        img = self.transform(img)
        output['image'] = img
        
        return output
    
   