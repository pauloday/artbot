from re import sub, escape
from yaml import load, FullLoader
from math import floor

# get the settings from the run yaml
def get_settings(run):
    return {
        'title': run.get('title') or 'Untitled',
        'images': run.get('images') or 20,
        'time_scale': run.get('time_scale') or 1,
        'noise_prompt_seeds': run.get('noise_prompt_seeds') or [],
        'noise_prompt_weights': run.get('noise_prompt_weights') or [],
        'size': run.get('size') or [730, 730],
        'init_image': run.get('init_image') or None,
        'init_weight': run.get('init_weight') or 0,
        'clip_model': run.get('clip_model') or 'ViT-B/32',
        'vqgan_config': run.get('vqgan_config') or 'vqgan_imagenet_f16_1024.yaml',
        'vqgan_checkpoint': run.get('vqgan_checkpoint') or 'vqgan_imagenet_f16_1024.ckpt',
        'step_size': run.get('step_size') or 0.05,
        'cutn': run.get('cutn') or 64,
        'cut_pow': run.get('cut_pow')or 1.,
        'seed': run.get('seed') or 0,
        'video': run.get('video') or False,
        'fps': run.get('fps') or 24
    }

# takes a 'n**prompt' and converts it to [prompt, prompt...]
# if there's no multiplier, just return [prompt]
mult_tok = '**'
def expand_iteration(line):
    if mult_tok in line:
        parts = line.split(mult_tok)
        n = int(parts[0])
        prompt = parts[1]
        return [prompt] * n
    return [line]

# lmao
def flatten_array(t):
    return [item for sublist in t for item in sublist]

# the yaml has any of the settings from above, and a prompts section
# the prompts section is an array of prompts, one per iteration
# iterations can be repeated with 'n**iteration prompt'
# there's also a image_prompts section that follows the same rules
# the text prompts are used for the iteration count
# returns (settings_dict, prompts_array, image_prompts)
def parse_yaml(yaml):
    parsed = load(yaml, Loader=FullLoader)
    settings = get_settings(parsed['settings'])
    prompts = flatten_array(map(expand_iteration, parsed['prompts']))
    image_prompts = [] #flatten_array(map(expand_iteration, parsed['image_prompts']))
    return (settings, prompts, image_prompts)
