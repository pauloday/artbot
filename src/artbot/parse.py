from re import search
from yaml import load, FullLoader
from math import floor

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

def parse_swap_part(part):
    pair = search(r'(.*)__(.*)', part)
    prompt_str = part
    num = 1.0
    if pair:
        groups = pair.groups()
        prompt_str = groups[0].strip()
        num = float(groups[1])
    return (prompt_str, num)

# parses 'one--two,2' for 300 iterations into {0.0: 'one', 100.0: 'two'}
sequence_token = '--'
def parse_swap(prompt, total_i):
    if '--' in prompt:
        parts = prompt.split(sequence_token)
        ratio_sum = 0
        pairs = []
        for part in parts:
            prompt_str, num = parse_swap_part(part)
            ratio_sum += num
            pairs.append((prompt_str, num))
        table = {}
        i = 0
        for prompt, num in pairs:
            table[i] = prompt
            i += floor(total_i*(num/ratio_sum))
        return table
    return prompt

concur_token = '||'
def parse_prompt(instr, total_i):
    if instr:
        prompts = [] # each input prompt can actually be many prompts
        if concur_token in instr:
            prompts = prompts + instr.split(concur_token)
        else:
            prompts.append(instr)
        prompts = list(map(lambda p: parse_swap(p, total_i), prompts))
        return prompts

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
        # try to set all the values
        new_args = {
            #TODO: refactor this to be less awkward
            'prompts': parse_prompt(run.get('prompt'), total_i),
            'image_prompts': parse_prompt(run.get('image_prompt'), total_i),
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
        for arg in args:
            if new_args[arg]:
                args[arg] = new_args[arg]
                
        runs[run_name] = args.copy()
    return title, runs