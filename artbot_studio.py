import sys, inspect, os, cv2
sys.path.append('./taming-transformers')
import streamlit as st
import util
import pprint
import Single
# this will pipe most output from colab to streamlit
def st_print(*args):
    strs = map(pprint.pformat, args)
    st.info(' '.join(strs))

if 'oldprint' not in __builtins__:
    __builtins__['oldprint'] = __builtins__['print']
__builtins__['print'] = st_print
state = st.session_state
# st.write(f'Device count: {util.dev_count()}')
side = st.sidebar
util.init_state_field(state, 'running', False)
state['running'] = st.button('Run')


util.init_state_field(state, 'args', {
    'prompts': [],
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
# write info and do run
if args:
    side.write('Loaded arguments:')
    side.write(args)
if batch_type == 'Single':
    batch = Single.Single(args)
if state['running']:
    top_status = st.empty()
    progress_bar = st.empty()
    bar = progress_bar.progress(0)
    image_box = st.empty()
    bottom_status = st.empty()
    
    top_status.write(f'Generating {args["prompts"]}...')
    batch.write_info()
    image_box = batch.run(bar, image_box, bottom_status)
    st.session_state['running'] = False

    top_status.write('Generating video...')
    image_folder = args['gallery']
    video_name = batch.title

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(str(video_name), 0, 30, (width,height))

    prog = 0
    for image in images:
        bar.progress(prog/len(images))
        video.write(cv2.imread(os.path.join(image_folder, image)))
        prog += 1

    cv2.destroyAllWindows()
    video.release()
    image_box.write(video)
    top_status.write(f'Run complete, output saved to {args["gallery"]}.')