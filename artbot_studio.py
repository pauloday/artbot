import sys, inspect
sys.path.append('./taming-transformers')
import streamlit as st
import util
import pprint
import Single
# this will pipe most output to streamlit, but not some progress bars
# I think I can reroute those but it might clobber the streamlit output
# pretty sure each progress bar update will be its own line
# so I'm leaning towards just having the colab page be the runner log
# def st_print(*args):
#     strs = map(pprint.pformat, args)
#     st.info(' '.join(strs))

# if 'oldprint' not in __builtins__:
#     __builtins__['oldprint'] = __builtins__['print']
# __builtins__['print'] = st_print
state = st.session_state
# st.write(f'Device count: {util.dev_count()}')
side = st.sidebar
util.init_state_field(state, 'running', False)
state['running'] = st.button('Run')


util.init_state_field(state, 'args', {
    'prompts': [],
    'image_prompts': None,
    'iterations': 1,
    'images_per_prompt': 1,
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
})

args = state.args

batch_type = st.selectbox(
    'Batch type',
    ('Single', 'more coming soon')
)
# set up a batch
args['iterations'] = side.number_input('Iterations', min_value=1, value=int(args['iterations']))
args['images_per_prompt'] = side.number_input('Images per prompt', min_value=1, value=int(args['images_per_prompt']))

args['size'][0] = side.number_input('Width', min_value=0, value=int(args['size'][0]))
args['size'][1] = side.number_input('Height', min_value=0, value=int(args['size'][1]))
    
if not state['running']:
    batch = None
    num_prompts = st.number_input('Number of prompts', min_value=1)
    args['prompts'] = util.prompts_form(args['prompts'], st, num_prompts)
else:  # we are running
    run_tag = 'Stop'
    st.write('See the colab tab or terminal output for more info')
# write info and do run
if args:
    side.write('Loaded arguments:')
    side.write(args)
if batch_type == 'Single':
    batch = Single.Single(args)
if state['running']:
    batch.write_info()
    batch.run(st)
    st.session_state['running'] = False
    st.write('Select a batch type to reset run')