import os
import torch
import argparse
import time
from diffusers import DiffusionPipeline

class Predictor:
    def __init__(self):
        self.pipe = self._load_model()

    def _load_model(self):
        model = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            custom_pipeline="latent_consistency_txt2img",
            custom_revision="main"
        )
        model.to(torch_device="cpu", torch_dtype=torch.float32).to('mps:0')
        return model

    def predict(self, prompt: str, width: int, height: int, steps: int, seed: int = None) -> str:
        seed = seed or int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        torch.manual_seed(seed)

        result = self.pipe(
            prompt=prompt, width=width, height=height,
            guidance_scale=8.0, num_inference_steps=steps,
            num_images_per_prompt=1, lcm_origin_steps=50,
            output_type="pil"
        ).images[0]

        return self._save_result(result, prompt, seed, width, height, steps)

    def _save_result(self, result, prompt: str, seed: int, width: int, height: int, steps: int):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Make prompt filesystem safe
        prompt = prompt.replace(" ", "_")
        prompt = prompt.replace("/", "_")
        prompt = prompt.replace("\\", "_")
        prompt = prompt.replace(":", "_")
        prompt = prompt.replace("*", "_")
        prompt = prompt.replace("?", "_")
        prompt = prompt.replace("\"", "_")
        prompt = prompt.replace("<", "_")
        prompt = prompt.replace(">", "_")
        prompt = prompt.replace("|", "_")

        output_path = os.path.join(output_dir, f"out-prompt-{prompt}-time-{timestamp}-seed-{seed}-width-{width}-height-{height}-steps-{steps}.png")
        result.save(output_path)
        return output_path

def main():
    args = parse_args()
    predictor = Predictor()

    # If prompt file is provided, read prompts from file
    if args.prompt_file:
        prompts = []
        with open(args.prompt_file, "r") as f:
            prompt_lines = f.readlines()
            prompts = [prompt.strip() for prompt in prompt_lines]

        if args.continuous:
            print("Continuous passed, will iterate forever.")

            try:
                while True:
                    for prompt in prompts:
                        print(f"Generating image for prompt: '{prompt}'")
                        output_path = predictor.predict(prompt, args.width, args.height, args.steps, args.seed)
                        print(f"Output image saved to: {output_path}")
            except KeyboardInterrupt:
                print("\nStopped by user.")
        else:
            for prompt in prompts:
                output_path = predictor.predict(prompt, args.width, args.height, args.steps, args.seed)
                print(f"Output image saved to: {output_path}")
    else:
        if not args.prompt:
            print("Please provide a prompt or a prompt file.")
            return

        if args.continuous:
            try:
                while True:
                    output_path = predictor.predict(args.prompt, args.width, args.height, args.steps, args.seed)
                    print(f"Output image saved to: {output_path}")
            except KeyboardInterrupt:
                print("\nStopped by user.")
        else:
            output_path = predictor.predict(args.prompt, args.width, args.height, args.steps, args.seed)
            print(f"Output image saved to: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images based on text prompts.")
    parser.add_argument("--prompt", type=str, default=None, help="A text prompt for image generation.")
    parser.add_argument("--prompt-file", type=str, default=None, help="A file containing text prompts for image generation, one prompt per line.")
    parser.add_argument("--width", type=int, default=512, help="The width of the generated image.")
    parser.add_argument("--height", type=int, default=512, help="The height of the generated image.")
    parser.add_argument("--steps", type=int, default=8, help="The number of inference steps.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random number generation.")
    parser.add_argument("--continuous", action='store_true', help="Enable continuous generation.")
    return parser.parse_args()

if __name__ == "__main__":
    main()
