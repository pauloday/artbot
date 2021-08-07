from re import search
from yaml import load, FullLoader

default_args = {
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
# parse the '||' and '--' tokens
# || - split into parrallel prompts
# -- - run the prompts one after another
# sequence  steps go into the runner as [prompt, ratio]
concur_token = '||'
sequence_token = '--'
def parse_prompt(instr):
    if instr:
        prompts = []
        if concur_token in instr:
            prompts = prompts + instr.split(concur_token)
        else:
            prompts.append(instr)
        def parse(prompt):
            if '--' in prompt:
                tups = prompt.split(sequence_token)
                parsed_prompt = []
                for tup_str in tups:
                    groups = search(r'(.*),(.*)', tup_str).groups()
                    tup = (groups[0].strip(), float(groups[1]))
                    parsed_prompt.append(tup)
                return parsed_prompt
            else:
                return prompt
        prompts = list(map(parse, prompts))
        return prompts    

# iterate over the keys
# for each key, parse into an args dictionary
# put that into a dictionary under the run name
def parse_yaml(yaml):
    parsed = load(yaml, Loader=FullLoader)
    runs = {}
    args = default_args
    title = parsed['title']
    del parsed['title']
    for run_name in parsed:
        run = parsed[run_name]
        # try to set all the values
        new_args = {
            'prompts': parse_prompt(run.get('prompts')),
            'image_prompts': parse_prompt(run.get('image_prompts')),
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
        for arg in args:
            if new_args[arg] != None:
                args[arg] = new_args[arg]
        runs[run_name] = args.copy()
    return title, runs