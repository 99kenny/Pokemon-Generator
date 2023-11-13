import os
import random

from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms

class PokemonDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, image_num=1, prompt_num=20, transform=None, Tokenizer = None):
        self.image_num = image_num
        self.prompt_num = prompt_num
        self.root_dir = root_dir
        self.transform = transform
        self.Tokenizer = Tokenizer

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
        
    def tokenize_captions(self, captions,is_train=True):
        inputs = self.Tokenizer(captions, max_length=self.Tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        tabular = self.tabular.iloc[idx].to_numpy()
        p_type = self.p_type.iloc[idx]
        prompt = self.prompts.iloc[idx]
        image = self.images[idx]
        sampled_img = random.sample(list(range(0,5)), self.image_num)
        
        output = dict()
        output['tabular'] = list(tabular)
        output['prompt'] = ','.join(random.sample(prompt.split(','), self.prompt_num))
        output['prompt'] = self.tokenize_captions(output['prompt'])
        output['p_type'] = int(p_type)
        output['image'] = []
        for index, file in enumerate(os.listdir(image)):
            if index in sampled_img:
                file_path = os.path.join(image,file)
                print(file_path)
                img = Image.open(file_path)
                img = self.transform(img)
                output['image'].append(img)
        return output
    
    
        
'''
if __name__ == '__main__':
    dataset = PokemonDataset(root_dir='C:/Users/pc03/Desktop/lora/dataset') 
    out = dataset.__getitem__(0)
    print(out['tabular'])
    print(out['prompt'])
    print(out['p_type'])
    print(out['image'])
    
'''

