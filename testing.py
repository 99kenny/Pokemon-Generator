
from load_data import PokemonDataset
from model import diffusion_model
from torch.utils.data import DataLoader
from utils import model_load, unnorm
import torch
from torchvision.utils import save_image
from datetime import datetime
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error

PTYPE_CATEGORY = {0: 'bug',
                1: 'dark',
                2: 'dragon',
                3: 'electric',
                4: 'fairy',
                5: 'fighting',
                6: 'fire',
                7: 'flying',
                8: 'ghost',
                9: 'grass',
                10: 'ground',
                11: 'ice',
                12: 'normal',
                13: 'poison',
                14: 'psychic',
                15: 'rock',
                16: 'steel',
                17: 'water'}

NEGATIVE_PROMPTS = "out of frame, extra fingers, mutated hands, monochrome, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, glitchy, bokeh, ((flat chested)), ((((visible hand)))), ((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), (((disfigured))), out of frame, ugly, (bad anatomy), gross proportions, (malformed limbs), (((extra legs))), mutated hands, (fused fingers), (too many fingers), multiple subjects, extra heads"

def test(args):
    inference_prompt = 'high resolution, masterpiece, best quality, ' + args.inference_prompt +', in style of pokemon,'# Focus and Sharpness: Make sure the image is focused and sharp and encourages the viewer to see it as a work of art printed on fabric.'
    if args.image_gen :
        model_base = args.pretrained_model_name_or_path
        pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.unet.load_attn_procs('output\pokemon\pytorch_lora_weights.safetensors')
        pipe.to("cuda")
        now = str(datetime.now())[:-7].replace(':', ' ')
        for i in range(10):
            output = pipe(inference_prompt, negative_prompt=NEGATIVE_PROMPTS, num_inference_steps=100, height = args.resolution, width = args.resolution)
            if output.nsfw_content_detected[0] == False :
                image = output.images[0]
                image.save(f'output/image/inference_img_{now}_{i}.jpg')

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
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = diffusion_model(args).to(device=device)
    model.set_lora(args)
    model = model_load(model, args)
    model.eval()
    
    dataset = PokemonDataset(root_dir=args.dataset_dir, args = args, transform=train_transforms, Tokenizer= model.tokenizer) 
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    scaler = dataset.scaler

    print(inference_prompt)
    inputs_ids = model.tokenizer(inference_prompt, max_length=model.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device = device)

    feature_pred, logit_pred = model.inference(input_ids=inputs_ids)
    unnormed_feature = scaler.inverse_transform(feature_pred[0].detach().cpu().numpy().reshape(1, -1))

    for key, value in zip(['weight_kg','height_m','attack','defense','sp_attack','sp_defense'], unnormed_feature[0]):
        print(f'{key} \t:  {(value)}')

    prediction_tabular = []
    target_tabular = []
    prediction_class = []
    target_class = []
    for step, batch in enumerate(tqdm(train_dataloader)):
        if not args.image_gen :
            input_ids = batch['prompt'][0].to(device = device)
            tabular = scaler.inverse_transform(batch['tabular'])
            feature_pred, logit_pred = model.inference(input_ids=input_ids)
            unnormed_feature = scaler.inverse_transform(feature_pred[0].detach().cpu().numpy().reshape(1, -1))
            pred = {}
            prediction_tabular.append(unnormed_feature[0])
            prediction_class.append(int(torch.argmax(logit_pred)))
            trg = {}
            target_tabular.append(tabular[0])
            target_class.append(batch['p_type'])
    assert len(prediction_tabular) == len(target_tabular)
    result = {}
    for key, value1, value2 in zip(['weight_kg','height_m','attack','defense','sp_attack','sp_defense'], np.array(prediction_tabular).T, np.array(target_tabular).T):
        result[key] = mean_squared_error(value1, value2)**0.5
    
    
    acc = [i for i,j in zip(prediction_class, target_class) if i == j]
    print(prediction_class)
    print(f'Pokemon_type_Acc : \n {len((acc)) / len(prediction_class)}')
