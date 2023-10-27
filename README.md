# Run latent consistency models on your Mac

Latent consistency models (LCMs) are based on Stable Diffusion, but they can
generate images much faster, needing only 4 to 8 steps for a good image
(compared to 25 to 50 steps).
[Simian Luo et al](https://arxiv.org/abs/2310.04378) released the first Stable
Diffusion distilled model. It’s distilled from the Dreamshaper fine-tune by
incorporating classifier-free guidance into the model’s input.

You can
[run Latent Consistency Models in the cloud on Replicate](https://replicate.com/luosiallen/latent-consistency-model),
but it's also possible to run it locally.

## Prerequisites

You’ll need:

- a Mac with an M1 or M2 chip
- 16GB RAM or more
- macOS 12.3 or higher
- Python 3.10 or above

## Install

Run this to clone the repo:

    git clone https://github.com/replicate/latent-consistency-model.git
    cd latent-consistency-model

Set up a virtualenv to install the dependencies:

    python3 -m pip install virtualenv
    python3 -m virtualenv venv

Activate the virtualenv:

    source venv/bin/activate

(You'll need to run this command again any time you want to run the script.)

Then, install the dependencies:

    pip install -r requirements.txt

## Run

The script will automatically download the
[`SimianLuo/LCM_Dreamshaper_v7`](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7)
(3.44 GB) and
[safety checker](https://huggingface.co/CompVis/stable-diffusion-safety-checker)
(1.22 GB) models from HuggingFace.

```sh
python main.py \
  --prompt "a beautiful apple floating in outer space, like a planet" \
  --steps 4 --width 512 --height 512
```

You’ll see an output like this:

```sh
Output image saved to: output/out-prompt-a_beautiful_apple_floating_in_outer_space,_like_a_planet-time-20231027-181911-seed-7445-width-512-height-512-steps-4.png
Using seed: 48404
100%|███████████████████████████| 4/4 [00:00<00:00,  5.54it/s]
```

### Run With A Prompt File

You can also run the script with a prompt file.

1. Create a prompt file with a few creative prompts:

```sh
echo "a beautiful apple floating in outer space, like a planet" > prompts.txt
echo "a knight in armor, with a sword and shield made of jelly" >> prompts.txt
echo "a cat with a human face" >> prompts.txt
```

2. Then run the script with the `--prompt-file` option:

```sh
python main.py \
  --prompt-file prompts.txt \
  --steps 4 --width 512 --height 512
```

You'll see output like this:

```sh
Using seed: 58067
100%|████████████████████████████████████| 4/4 [00:01<00:00,  3.28it/s]
Output image saved to: output/out-prompt-A_steampunk_city_at_sunset-time-20231027-181942-seed-58067-width-512-height-512-steps-4.png
Using seed: 13995
100%|████████████████████████████████████| 4/4 [00:00<00:00,  5.19it/s]
Output image saved to: output/out-prompt-A_dragon_playing_chess_with_a_knight-time-20231027-181943-seed-13995-width-512-height-512-steps-4.png
```

## Options

| Parameter     | Type | Default | Description                           |
| ------------- | ---- | ------- | ------------------------------------- |
| --prompt      | str  | N/A     | A text string for image generation.   |
| --prompt-file | str  | N/A     | A text file with one prompt per-line. |
| --width       | int  | 512     | The width of the generated image.     |
| --height      | int  | 512     | The height of the generated image.    |
| --steps       | int  | 8       | The number of inference steps.        |
| --seed        | int  | None    | Seed for random number generation.    |
| --continuous  | flag | False   | Enable continuous generation.         |
