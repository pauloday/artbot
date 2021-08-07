from re import search
from yaml import load

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
    parsed = load(yaml)
    runs = {}
    args = default_args
    for run in parsed:
        # try to set all the values
        new_args = {
            'prompts': parse_prompt(run.prompt),
            'image_prompts': parse_prompt(run.image_prompt),
            'iterations': run.iterations,
            'images_per_prompt': run.images,
            'noise_prompt_seeds': run.noise_prompt_seeds,
            'noise_prompt_weights': run.noise_prompt_weights,
            'size': run.size,
            'init_image': run.init_image,
            'init_weight': run.init_weight,
            'clip_model': run.clip_model,
            'vqgan_config': run.vqgan_config,
            'vqgan_checkpoint': run.vqgan_checkpoint,
            'step_size': run.step_size,
            'cutn': run.cutn,
            'cut_pow': run.cut_pow,
            'seed': run.seed,
        }
        # only set the ones that actually were set
        # this is how the carry over feature is implemented
        for arg in args:
            if arg in new_args:
                args[arg] = new_args[arg]
        runs[run] = args
    return runs