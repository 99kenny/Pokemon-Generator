
from model import diffusion_model
from utils import model_load, unnorm
import torch
from torchvision.utils import save_image
from datetime import datetime
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

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
    inference_prompt = '<cls>, high resolution, masterpiece, best quality' + args.inference_prompt +', in style of pokemon,'# Focus and Sharpness: Make sure the image is focused and sharp and encourages the viewer to see it as a work of art printed on fabric.'
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
    
    model = diffusion_model(args)
    model.set_lora(args)
    model = model_load(model, args)
    model.eval()
    inputs_ids = model.tokenizer(inference_prompt, max_length=model.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
    feature_pred, logit_pred = model.inference(input_ids=inputs_ids)
    unnormed_feature = unnorm(feature_pred[0], args)
    for key, value in zip(['weight_kg','height_m','attack','defense','sp_attack','sp_defense'], unnormed_feature):
        print(f'{key} \t:  {(value)}')
    print(PTYPE_CATEGORY[int(torch.argmax(logit_pred[0]))])
