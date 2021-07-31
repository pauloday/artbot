import torch
import argparse


def windows_path_sanitize(string):
    return ''.join(i for i in string if i not in ':*?<>|')

def dev_count():
    return torch.cuda.device_count()

def init_state_field(state, field, default):
    if field not in state:
        state[field] = default

def prompts_form(prompts, form, num_prompts):
    form.write('Use tuples to split iteration time between prompts')
    form.write('E.G (river, 1), (lava, 1) will do half iterations on river and half on lava')
    form.write('You can also do multiple prompts, concurrently')
    for i in range(int(num_prompts)):
        if i >= len(prompts):
            prompts.append(None)
        prompts[i] = form.text_input(f'Prompt #{i}', value=prompts[i])
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