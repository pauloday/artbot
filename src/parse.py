from re import sub, escape
from yaml import load, FullLoader
from math import floor
from parse_prompt import parse_prompt

def get_default_args():
    return {
        'prompt': [],
        'image_prompt': None,
        'iterations': 200,
        'images_per_prompt': 30,
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

# You can reference other prompts and image outputs with '[ref]':
# prompt, image_prompt, init_image: '[prior_run], [predefined_prompt]'
# predefined prompts are fields in the object root that only have a string
# { title: 'Woah', neon_prompt: 'Billowing Neon Lights', neon_run: { prompt: '[neon_prompt]' } }
# Using a prior run will replace the the prompt or image output with the value from that run
# prompt updates happen first, image linking implemented in artbot.py

ref_reg = lambda name: escape(f'[{name}]')
# 'regular prompt except with a [ref] in it', ref, update
# -> 'regular prompt except with a update in it'
def update_ref(prompt, name, update):
    return sub(ref_reg(name), update, prompt)

# update all refs inside a dict of prompt frames
def update_refs(prompts_dict, name, update):
    for i, prompts in prompts_dict.items():
        up_ref = lambda p: update_ref(p, name, update)
        prompts_dict[i] = list(map(up_ref, prompts))
    return prompts_dict

# updates a prompt using a set of runs
# run is a dict of dicts with a 'prompt' key
# the prompt is a string, this is run before parse_prompt
def update_prompt_ref(prompt, runs):
    for name, run in runs.items():
        prompt = update_ref(prompt, name, run['prompt'])
    return prompt

def deref_prompts(runs):
    def update_name(name, parent):
        prompt = parent[name]['prompt']
        derefed = update_prompt_ref(prompt, runs)
        parent[name]['prompt'] = derefed
    #todo: use a real parser generator + grammer
    for name in runs:
        update_name(name, runs)
    return runs

# iterate over the keys
# for each key, parse into an args dictionary
# put that into a dictionary under the run name
def parse_yaml(yaml):
    parsed = load(yaml, Loader=FullLoader)
    runs = {}
    args = get_default_args()
    title = parsed['title']
    del parsed['title']
    # section_names will be for one of:
    # run (dict)
    # solo prompt run (str)
    for section_name in parsed:
        run = parsed[section_name]
        # implement "title: prompt" shorthand
        if type(run) == str:
            prompt = run
            run = args.copy()
            run['prompt'] = prompt
        new_args = {
            #TODO: refactor this to be less awkward?
            'prompt': run.get('prompt'),
            'image_prompt': run.get('image_prompt'),
            'iterations': run.get('iterations'),
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
            elif new_args[arg] == False:
                args[arg] = get_default_args()[arg]
        runs[section_name] = args.copy()
    # update all of the prompt references
    runs = deref_prompts(runs)
    for name, r in runs.items():
        runs[name]['prompt'] = parse_prompt(r['prompt'], r['iterations'])
        if runs[name]['image_prompt']:
            runs[name]['image_prompt'] = parse_prompt(r['image_prompt'], r['iterations'])
    return title, runs

# print(parse_yaml(open('demo.yml').read()))