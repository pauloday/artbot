from re import search
from yaml import load, FullLoader
from math import floor
from parse_prompt import parse_prompt

def get_default_args():
    return {
        'prompts': [],
        'image_prompts': None,
        'iterations': 300,
        'images_per_prompt': 10,
        'noise_prompt_seeds': [],
        'noise_prompt_weights': [],
        'size': [1000, 500],
        'init_image': None, # this can be None
        'init_weight': 0.,
        'clip_model': 'ViT-B/32',
        'vqgan_config': 'vqgan_imagenet_f16_1024.yaml',
        'vqgan_checkpoint': 'vqgan_imagenet_f16_1024.ckpt',
        'step_size': 0.05,
        'cutn': 64,
        'cut_pow': 1.,
        'seed': 0,
    }

# iterate over the keys
# for each key, parse into an args dictionary
# put that into a dictionary under the run name
def parse_yaml(yaml):
    parsed = load(yaml, Loader=FullLoader)
    runs = {}
    args = get_default_args()
    title = parsed['title']
    del parsed['title']
    for run_name in parsed:
        run = parsed[run_name]
        total_i = run.get('iterations') or args['iterations']
        image_prompt = run.get('image_prompt') or None
        prompt = run.get('prompt') or None
        # try to set all the values
        new_args = {
            #TODO: refactor this to be less awkward
            'prompts': prompt and parse_prompt(run.get('prompt'), total_i),
            'image_prompts': image_prompt and parse_prompt(image_prompt, total_i),
            'iterations': total_i,
            'images_per_prompt': run.get('images'),
            'noise_prompt_seeds': run.get('noise_prompt_seeds'),
            'noise_prompt_weights': run.get('noise_prompt_weights'),
            'size': run.get('size'),
            'init_image': run.get('init_image'),
            'init_weight': run.get('init_weight'),
            'clip_model': run.get('clip_model'),
            'vqgan_config': run.get('vqgan_config'),
            'vqgan_checkpoint': run.get('vqgan_checkpoint'),
            'step_size': run.get('step_size'),
            'cutn': run.get('cutn'),
            'cut_pow': run.get('cut_pow'),
            'seed': run.get('seed')
        }
        # only set the ones that actually were set
        # this is how the carry over feature is implemented
        # if explicitly false reset arg
        for arg in args:
            if new_args[arg]:
                args[arg] = new_args[arg]
                
        runs[run_name] = args.copy()
    return title, runs

print(parse_yaml(open('demo.yml').read()))