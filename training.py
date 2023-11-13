import torch.functional as F 
from torch.utils.data import DataLoader

from model import diffusion_model
from load_data import PokemonDataset
from torchvision import transforms



def train(args):
    model = diffusion_model(args)
    

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
    for step, batch in enumerate(train_dataloader):
        
        print(type(batch['image']))
        print(batch['prompt'].shape)
        print(batch['p_type'].shape)
        print(batch['tabular'].shape)
        break
        #get_data
        pixel_values = None
        input_ids = None

        model_pred, target = model(pixel_values, input_ids)

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    pass