import os
import torch
import argparse
from cog import BasePredictor
import time
from diffusers import DiffusionPipeline
from latent_consistency_img2img import LatentConsistencyModelImg2ImgPipeline
from PIL import Image

class Predictor(BasePredictor):
    def setup(self):
        self.txt2img = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            custom_pipeline="latent_consistency_txt2img",
            custom_revision="main"
        )
        self.txt2img.to(torch_device="cpu", torch_dtype=torch.float32).to('mps:0')
        self.img2img = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", custom_pipeline="latent_consistency_img2img.py").to('mps:0')


    def predict(self, prompt: str, width: int, height: int, steps: int, seed: int = None, image: str = None) -> str:
        seed = seed or int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        torch.manual_seed(seed)

        if image:   
            print("img2img")
            input_image=Image.open(image)
            strength = 0.75
            result = self.img2img(
                image=input_image, prompt=prompt, width=width, height=height,
                guidance_scale=8.0, num_inference_steps=steps, strength=strength,
                num_images_per_prompt=1, lcm_origin_steps=50,
                output_type="pil"
            ).images[0]
        else:
            print("txt2img")
            result = self.txt2img(
                prompt=prompt, width=width, height=height,
                guidance_scale=8.0, num_inference_steps=steps,
                num_images_per_prompt=1, lcm_origin_steps=50,
                output_type="pil"
            ).images[0]

        return self._save_result(result)

    def _save_result(self, result):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = f"out-{timestamp}.png"
        result.save(output_path)
        return output_path

def main():
    args = parse_args()
    predictor = Predictor()
    predictor.setup()

    if args.continuous:
        try:
            output_path = args.image
            while True:
                output_path = predictor.predict(args.prompt, args.width, args.height, args.steps, args.seed, output_path)
                print(f"Output image saved to: {output_path}")
                f = open('index.html','w')
                f.write('<img src="'+output_path+'">')
                f.close()
                
        except KeyboardInterrupt:
            print("\nStopped by user.")
    else:
        output_path = predictor.predict(args.prompt, args.width, args.height, args.steps, args.seed)
        print(f"Output image saved to: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images based on text prompts.")
    parser.add_argument("prompt", type=str, help="A single text prompt for image generation.")
    parser.add_argument("--width", type=int, default=512, help="The width of the generated image.")
    parser.add_argument("--height", type=int, default=512, help="The height of the generated image.")
    parser.add_argument("--steps", type=int, default=8, help="The number of inference steps.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random number generation.")
    parser.add_argument("--image", type=str, default=None, help="The path to the image to be used as a base.")
    parser.add_argument("--continuous", action='store_true', help="Enable continuous generation.")
    parser.add_argument("--html", action='store_true', help="store html page.")
    return parser.parse_args()

if __name__ == "__main__":
    main()
