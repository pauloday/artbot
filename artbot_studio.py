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

util.init_state_field('args', state, {
    'prompts': [''],
    'image_prompts': None,
    'iterations': 300,
    'images_per_prompt': 100,
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
submitted = form.form_submit_button('Run')
if submitted:
    state['running'] = True
args['prompts'], title = util.prompts_form(form)
args['iterations'] = form.number_input('Iterations', min_value=1, value=int(args['iterations']))
args['images_per_prompt'] = form.number_input('Images per prompt', min_value=1, value=int(args['images_per_prompt']))
args['seed'] = form.number_input('Seed (adjust to get different versions)', min_value=0, value=args['seed'])
args['step_size'] = form.number_input('step size (how much each step changes the image)', min_value=0.0001, value=0.05)
args['cutn'] = form.number_input('Cutouts (low - less coherent, high - slower)', min_value=1, max_value=128, value=args['cutn'])

args['size'][0] = form.number_input('Width', min_value=0, value=int(args['size'][0]))
args['size'][1] = form.number_input('Height', min_value=0, value=int(args['size'][1]))


if not state['running']:
    # streamlit magic command, this will be parsed as markdown
    '''
    # Artbot Studio
    Welcome to Artbot! Enter a prompt to get started. The image size is tuned for Colab, but the other settings can be changes as you wish.
    '--' seperated pairs to switch the prompt midway through a run. The second value is the ratio of time to spend on that prompt.
    E.G `river, 1--lava, 1` will do half iterations on river and half on lava.

    You can also run prompts concurrently with '||'. This can be combined with the switching prompts to make complex prompts:
    `river, 1--lava, 1||ocean waves`
    
    If you do too many images per prompt the previews may stop displaying. The images and video should stll be saved though, you can see them in file browser in the terminal/Colab tab.
    
    Here's some cool prompts to start with:
    - `Multiple`
    - `Harmony`
    - `Dynamic`
    - `Parallel`
    - `Concurrent`
    - `Industrial`
    - `Simple`
    - `Flow`
    - `Outer space`
    - `fire lava, 1--mountain water, 1--ocean waves, 1`
    - `sunrise sunset horizon, 1--ocean, 2--forest, 3`
    - `mountains, 1--bright sky, 1||multiple, 1--dynamic, 2--frothy, 3||ocean waves trending on artstation`

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
    I just put them on the end. For example `Dynamic by Van Gogh trending on Artstation vray`
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
        batch = Single.Single(clean_args, title)

    # this will pipe most output from colab to streamlit
    def st_print(*args):
        strs = map(pprint.pformat, args)
        st.write(' '.join(strs))
    if 'oldprint' not in __builtins__:
        __builtins__['oldprint'] = __builtins__['print']
    __builtins__['print'] = st_print
    video_box = st.empty()
    top_status = st.empty()
    top_status.write(f'Generating {args["prompts"]}...')
    update_box = st.container()
    '''
    Image preview is rate limited, it will only update once every 5 seconds.
    If the image dimensions are too high, you'll get an out of memory error.
    When this happens you'll have to go back to the Colab tab, restart the runtime, and re-run the last 2 cells.
    Otherwise you'll just run out of memory on all runs.
    '''

    # set up so frames can be added as they're generated
    # the output dir may not be only for this run, so this is the best way
    gallery = batch.args['gallery']
    frames = []
    def add_frame(frame):
        frames.append(frame)

    # do the run
    batch.write_info()
    st.sidebar.write(batch.args)
    batch.run(update_box, add_frame)
    st.session_state['running'] = False


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