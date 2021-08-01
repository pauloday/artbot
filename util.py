import torch
import os
import argparse
from ast import literal_eval

def windows_path_sanitize(string):
    return ''.join(i for i in string if i not in ':*?<>|')

def dev_count():
    return torch.cuda.device_count()

def init_state_field(field, state, default):
    if field not in state:
        state[field] = default

# just display existing prompts
def prompts_form(num_prompts, form):
    prompts = []
    for i in range(num_prompts):
        if len(prompts) >= i:
            prompts.append('')
        prompt = form.text_input(f'Prompt #{i + 1}', value=prompts[i])
        # this was loaded from state and is a ratio set, parse to the semicolon list
        if len(prompt) > 0 and prompt[0] == '[':
            prompt = ''
            parsed = literal_eval(prompts[i])
            for tup in parsed:
                print(prompt + f'{tup}; ')
                prompt = prompt + (f'{tup}; ')
        if len(prompt) > 0 and prompt[0] == '(':
            tups = prompt.split(';')
            parsed_prompt = []
            for tup in tups:
                parsed_prompt.append(literal_eval(tup.strip()))
            prompt = parsed_prompt
        prompts[i] = prompt
    return prompts

def state_to_args(state):
    return argparse.Namespace(
        prompts=state['prompts'],
        image_prompts=[],
        iterations=state['iterations'],
        images_per_prompt=state['images_per_prompt'],
        noise_prompt_seeds=[],
        noise_prompt_weights=[],
        size=state['size'],
        init_image=None,
        init_weight=0.,
        clip_model='ViT-B/32',
        vqgan_config='vqgan_imagenet_f16_1024.yaml',
        vqgan_checkpoint='vqgan_imagenet_f16_1024.ckpt',
        step_size=0.05,
        cutn=64,
        cut_pow=1.,
        seed=0,
    )