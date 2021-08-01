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
state['running'] = st.sidebar.button('Run')

util.init_state_field('args', state, {
    'prompts': [''],
    'image_prompts': None,
    'iterations': 300,
    'images_per_prompt': 80,
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
form.form_submit_button('Update (stops run)')
args['iterations'] = form.number_input('Iterations', min_value=1, value=int(args['iterations']))
args['images_per_prompt'] = form.number_input('Images per prompt', min_value=1, value=int(args['images_per_prompt']))

args['size'][0] = form.number_input('Width', min_value=0, value=int(args['size'][0]))
args['size'][1] = form.number_input('Height', min_value=0, value=int(args['size'][1]))
num_prompts = form.number_input('Number of prompts', min_value=1, value=1)
args['prompts'] = util.prompts_form(num_prompts, form)

if not state['running']:
    # streamlit magic command, this will be parsed as markdown
    '''
    # Artbot Studio
    Double semicolon seperated pairs to switch the prompt midway through a run. The second value is the ratio of time to spend on that prompt.
    E.G `river, 1;;lava, 1` will do half iterations on river and half on lava.
    
    If you do too many images per prompt the previews may stop displaying. The images and video should stll be saved though, you can see them in file browser in the terminal/Colab tab.
    
    Here's some cool prompts to start with:
    - `Harmony`
    - `Dynamic`
    - `Multiple`
    - `Parallel`
    - `Concurrent`
    - `Industrial`
    - `fire lava, 1;;mountain water, 1;;ocean waves', 1`
    - `sunrise sunset horizon, 1;;ocean, 2;;forest, 3`

    You can also add artist styles using `by` or `in the style of`, for example `Dynamic by Van Gogh`.
    Here's some good ones:
    - `Van Gogh`
    - `Salvador Dali`
    - `M.C. Escher`
    - `Claude Monet`
    - `Alex Grey`
    - `Thomas Moran`
    - `Studio Ghibli`
    - `Odilon Redon`

    I call these render strings. The model was trained with images from online art boards like Artstation, so these strings will make it more realistic/artsy.
    I just put them on the end, or use a | to seperate them. For example `Dynamic by Van Gogh trending on Artstation | vray`
    - `Artstation`
    - `Trending on Artstation`
    - `ArtstationHQ`
    - `Unreal Engine`
    - `vray`
    - `photorealistic`
    - `photo`
    - `painting`
    - `oil painting`
    - `matte painting`
    '''
if state['running'] and args:
# write info and do run
    clean_args = args.copy()
    if clean_args['prompts'][0] == '':
        del clean_args['prompts'][0]
    args = clean_args
    if batch_type == 'Single':
        title = util.windows_path_sanitize(str(args['prompts']))
        batch = Single.Single(clean_args, title)

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
    argv = ['-r', '20', '-f', 'image2', '-i', f'{tmp_dir}/%04d.jpg', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_name]
    ffpb.main(argv, tqdm=stqdm)
    shutil.rmtree(tmp_dir)

    video_box.video(str(video_name))
    top_status.write(f'Run complete, output saved to {batch.args["gallery"]}.')

if args:
    side.write('Loaded arguments:')
    side.write(args)