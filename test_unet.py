from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor, AttnProcessor2_0
import torch 

model_id = "runwayml/stable-diffusion-v1-5"
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda")
unet.set_attn_processor(AttnProcessor())

noisy_latents = torch.randn(1, 4, 64, 64).to("cuda")
timesteps = torch.randint(1, 100, size=(noisy_latents.shape[0], )).to("cuda")
timesteps = timesteps.long()
encoder_hidden_states = torch.randn(1, 77, 768).to("cuda")

model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
print(model_pred.shape)