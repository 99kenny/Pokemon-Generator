# ------------------------------------------
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------
# Modification:
# -- Joohyung Park - hynciath51@gmail.com, 
# -- Sangyun Lee - 99sansan@naver.com
# ------------------------------------------
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import torch
import torch.nn.functional as F 
from torch.utils.data import DataLoader

from model import diffusion_model
from load_data import PokemonDataset
from torchvision import transforms

def train(args):
    model = diffusion_model(args)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'device : {device}')
    # Data loader
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.RandomHorizontalFlip(), #if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset = PokemonDataset(root_dir=args.dataset_dir, transform=train_transforms, Tokenizer= model.tokenizer) 
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model.unet.requires_grad_(False)
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)

    model.set_lora(args)
    
    weight_dtype = torch.float32
    model.to(device, dtype=weight_dtype)



    for epoch in range(0,args.epochs):
        model.unet.train()
        train_loss = 0.0        
        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["image"].to(device = device, dtype=weight_dtype)
            input_ids = batch['prompt'].to(device = device)
            model_pred, target = model(pixel_values, input_ids)
            break
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        break
        pass