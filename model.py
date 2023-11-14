import torch
from torch import nn 

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers

class diffusion_model(nn.Module):
    def __init__(self, args):
        super(diffusion_model, self).__init__()

        self.tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
        self.unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.regressor = nn.Linear(in_features=args.latent_dim, out_features=args.num_features, bias=True)
        self.classifier = nn.Sequential(nn.Linear(args.latent_dim, args.latent_dim // 2),nn.Linear(args.latent_dim//2, args.num_classes),nn.Softmax(dim=1))
        self.lora_layers = None

    def set_lora (self, args):
        lora_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=args.rank,
            )

        self.unet.set_attn_processor(lora_attn_procs)    
        self.lora_layers = AttnProcsLayers(self.unet.attn_processors)


    def forward(self, pixel_values, input_ids):
        # Convert images to latent space
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = self.text_encoder(input_ids)[0]
        
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        flatten_latents = latents.reshape(latents.shape[0], -1)
        feature_pred = self.regressor(flatten_latents)
        logit_pred = self.classifier(flatten_latents)
        return model_pred, target, feature_pred, logit_pred
    
    def inference(self, input_ids):
        
        pass
        
    
    