
from model import diffusion_model
from utils import model_load

def test(args):
    model = diffusion_model(args)
    model.set_lora(args)
    model = model_load(model, args)
    print(model)