import os
import random

from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

class PokemonDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, image_num=5, prompt_num=20, transform=None):
        self.image_num = image_num
        self.prompt_num = prompt_num
        self.root_dir = root_dir
        self.transform = transform

        self.tabular = pd.read_csv(f'{root_dir}/pokemon_preprocessed.csv')
        self.p_type = self.tabular['type1'].astype('category').cat.codes
        self.tabular = self.tabular.drop(['type1'], axis=1)
        
        self.prompts = pd.read_csv(f'{root_dir}/pokemon_prompts.csv')
        self.prompts = self.prompts['prompts'].str.replace("'","").str.replace('[','').str.replace(']','')

        image_dir = os.path.join(root_dir, 'images')
        self.images = []
        for folder in os.listdir(image_dir):
            file_path = os.path.join(image_dir,folder)
            if len(file_path) == 0:
                raise ValueError
            self.images.append(file_path)    
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        tabular = self.tabular.iloc[idx].to_numpy()
        p_type = self.p_type.iloc[idx]
        prompt = self.prompts.iloc[idx]
        image = self.images[idx]
        image_idx = list(range(0,10))
        sampled_img = random.sample(image_idx, self.image_num)
        
        output = dict()
        output['tabular'] = tabular
        output['prompt'] = ','.join(random.sample(prompt.split(','), self.prompt_num))
        output['p_type'] = p_type
        output['image'] = []
        for index, file in enumerate(os.listdir(image)):
            if index in sampled_img:
                file_path = os.path.join(image,file)
                img = Image.open(file_path)
                output['image'].append(img)
        return output

if __name__ == '__main__':
    dataset = PokemonDataset(root_dir='C:/Users/pc03/Desktop/lora/dataset') 
    out = dataset.__getitem__(0)
    print(out['tabular'])
    print(out['prompt'])
    print(out['p_type'])
    print(out['image'])
    
