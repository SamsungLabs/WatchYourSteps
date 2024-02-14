import torch
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPix2PixZeroPipeline, StableDiffusionDiffEditPipeline
import os
import json
from diffusers.utils import load_image

os.environ['CURL_CA_BUNDLE'] = ''

sd_model_ckpt = "stabilityai/stable-diffusion-2-1"
# sd_model_ckpt = "runwayml/stable-diffusion-v1-5"
pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
    sd_model_ckpt,
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()
generator = torch.manual_seed(0)


img_url = "/user/a.mirzaei/in2n/instruct-pix2pix/outputs/3_s=100_r=512_tao=0.4_si=0.8_st=7.5_in.png"
meta_url = img_url.replace("_in.png", ".txt")
raw_image = load_image(img_url).convert("RGB")
# raw_image = raw_image.resize((768, 768))

with open(meta_url, 'r') as f:
  meta = json.load(f)
source_prompt = meta["input_prompt"]
# source_prompt = ""
target_prompt = meta["output_prompt"]



# img_url = "https://github.com/Xiang-cd/DiffEdit-stable-diffusion/raw/main/assets/origin.png"
# raw_image = load_image(img_url).convert("RGB").resize((512, 512))
# source_prompt = "a bowl of fruits"
# target_prompt = "a basket of fruits"


inpaint_strength = 0.8
steps = 20

with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False, dtype=torch.float16):

    mask_image = pipeline.generate_mask(
        image=raw_image,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        generator=generator,
        num_inference_steps=steps,
    )

    print(mask_image.sum(), mask_image.mean())

    inv_latents = pipeline.invert(prompt=source_prompt,
                                image=raw_image, 
                                generator=generator,
                                inpaint_strength=inpaint_strength,
                                num_inference_steps=steps,
    ).latents

    image = pipeline(
        prompt=target_prompt,
        mask_image=mask_image,
        image_latents=inv_latents,
        generator=generator,
        negative_prompt=source_prompt,
        inpaint_strength=inpaint_strength,
        num_inference_steps=steps,
    ).images[0]
    image.save("test_diffedit.png")