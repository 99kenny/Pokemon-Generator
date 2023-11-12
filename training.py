from model import diffusion_model
import torch.functional as F 

def tokenize_captions(captions,tokenizer,  is_train=True, ):
    captions = []
    inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")

    return inputs.input_ids

def train(args):
    model = diffusion_model(args)
    print(model)

    # Data loader
    train_dataloader = None
    
    model.unet.requires_grad_(False)
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)

    model.set_lora(args)
    for step, batch in enumerate(train_dataloader):
        #get_data
        pixel_values = None
        input_ids = None

        model_pred, target = model(pixel_values, input_ids)

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    pass