import sys, inspect, os, cv2
sys.path.append('./taming-transformers')
import streamlit as st
import util
import pprint
import Single
import math
import time
from stqdm import stqdm
from pathlib import Path
import shutil
import ffpb

state = st.session_state
# st.write(f'Device count: {util.dev_count()}')
side = st.sidebar
util.init_state_field('running', state, False)
state['running'] = st.button('Run')


util.init_state_field('args', state, {
    'prompts': [''],
    'image_prompts': None,
    'iterations': 100,
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
})

args = state.args

batch_type = 'Single'
#= st.sidebar.selectbox(
#     'Batch type',
#     ('Single', 'more coming soon')
# )
# set up a batch
form = st.sidebar.form(key='side_form')
args['iterations'] = form.number_input('Iterations', min_value=1, value=int(args['iterations']))
args['images_per_prompt'] = form.number_input('Images per prompt', min_value=1, value=int(args['images_per_prompt']))

args['size'][0] = form.number_input('Width', min_value=0, value=int(args['size'][0]))
args['size'][1] = form.number_input('Height', min_value=0, value=int(args['size'][1]))
num_prompts = form.number_input('Number of prompts', min_value=1, value=1)
args['prompts'] = util.prompts_form(num_prompts, form)
form.form_submit_button('Update (stops run)')

if not state['running']:
    st.write('Use semicolon separated tuples to split iteration time between prompts, the second value is the ratio of time to spend on that prompt.')
    st.write('E.G (\'river\', 1); (\'lava\', 1) will do half iterations on river and half on lava. You have to use quotes around the prompts in the ratio sets.')
    st.write('You can also do multiple prompts, concurrently. Individual prompts don\'t need quotes.')
if state['running'] and args:
# write info and do run
    clean_args = args.copy()
    if clean_args['prompts'][0] == '':
        del clean_args['prompts'][0]
    args = clean_args
    if batch_type == 'Single':
        batch = Single.Single(clean_args, title='tmp')
        # this will pipe most output from colab to streamlit
    def st_print(*args):
        strs = map(pprint.pformat, args)
        st.write(' '.join(strs))

    if 'oldprint' not in __builtins__:
        __builtins__['oldprint'] = __builtins__['print']
        __builtins__['print'] = st_print
    st.session_state['running'] = False
    video_box = st.empty()
    top_status = st.empty()
    top_status.write(f'Generating {args["prompts"]}...')
    update_box = st.beta_container()

    # set up so frames can be added as they're generated
    # the output dir may not be only for this run, so this is the best way
    gallery = batch.args['gallery']
    frames = []
    def add_frame(frame):
        frames.append(frame)

    # do the run
    batch.write_info()
    batch.run(update_box, add_frame)

    # copy images into a tmp dir with names set up for ffmpeg
    tmp_dir = f'{gallery}/tmp'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    i = 0
    for frame in frames:
        shutil.copyfile(frame, f'{tmp_dir}/{str(i).zfill(4)}.jpg')
        i += 1
    video_name = f'{gallery}/{math.floor(time.time())}.mp4'

    # ffpb uses some fancy print stuff, so put old print back
    __builtins__['print'] = __builtins__['oldprint']
    argv = ['-r', '30', '-f', 'image2', '-i', f'{tmp_dir}/%04d.jpg', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_name]
    ffpb.main(argv, tqdm=stqdm)
    shutil.rmtree(tmp_dir)

    video_box.video(str(video_name))
    top_status.write(f'Run complete, output saved to {batch.args["gallery"]}.')

if args:
    side.write('Loaded arguments:')
    side.write(args)