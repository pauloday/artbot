import sys
sys.path.append('./taming-transformers')
sys.path.append('./templates')
import streamlit as st
import pprint
import math
import time
from stqdm import stqdm
from parse import parse_yaml
from BatchRunner import BatchRunner
from pathlib import Path
from runner import run_args
import shutil
from streamlit_ace import st_ace
import blender

from yaml import dump

state = st.session_state
def init_state_field(field, state, default):
    if field not in state:
        state[field] = default

# st.write(f'Device count: {util.dev_count()}')
side = st.sidebar
init_state_field('running', state, False)
init_state_field('yaml', state, 'title: Studio Batch\n')

arg_docs = {
    'prompts': 'Prompt',
    'image_prompts': 'Image prompts',
    'iterations': 'Iterations',
    'images_per_prompt': 'Images per run',
    'noise_prompt_seeds': 'Noise prompt seeds',
    'noise_prompt_weights': 'Noise prompt weights',
    'size': 'Size, higer goes OOM. Colab maxes at around 1000x500 or 680x680',
    'init_image': 'Can be *ref, file path or URL',
    'init_weight': 'Init weight',
    'clip_model': 'Clip model',
    'vqgan_config': 'VQGAN config',
    'vqgan_checkpoint': 'VQGAN checkpoint',
    'step_size': 'Step size, how much each iteration changes the image',
    'cutn': 'Cutouts, add blind spots to force more coherance. Higher goes OOM',
    'cut_pow': 'Cutout size, higher goes OOM fast',
    'seed': 'Rng seed, adjust for different versions',
}
running = st.sidebar.button('Run')
if running:
    state['running'] = running
args = {}
st.sidebar.write('Only one template for now, more coming soon.')
form = st.sidebar.form(key='side_form')
template_dict = blender.generate(form)
submitted = form.form_submit_button('Add to config')
if submitted:
    state['yaml'] += '\n' + dump(template_dict, sort_keys=False)
'''
# Artbot Studio
'''
def draw_editor():
    state['yaml'] = st_ace(
        value=state['yaml'],
        language='yaml',
        tab_size='2',
        show_gutter=True,
        auto_update=False,
        readonly=state['running']
    )
if not state['running']:
    draw_editor()
else:
    top_status = st.empty()
    prog_box = st.container()
    image_box = st.empty()
    bot_status = st.empty()
    '''
    Image preview is rate limited, it will only update once every 5 seconds.
    If the image dimensions are too high, you'll get an out of memory error.
    When this happens you'll have to go back to the Colab tab, restart the runtime, and re-run the last 2 cells.
    Otherwise you'll just run out of memory on all runs.
    '''
    gallery_box = st.container()
    draw_editor()
    # this will pipe most output from colab to streamlit
    def st_print(*args):
        strs = map(pprint.pformat, args)
        bot_status.write(' '.join(strs))
    if 'oldprint' not in __builtins__:
        __builtins__['oldprint'] = __builtins__['print']
    __builtins__['print'] = st_print
    # rate as in only display a image every n seconds
    class ImageWriter():
        def __init__(self, rate, writer):
            self.rate = rate
            self.out_stamp = math.floor(time.time())
            self.writer = writer
        
        def write(self, image):
            now = math.floor(time.time())
            if now - self.out_stamp > self.rate:
                self.out_stamp = now
                self.writer(image)
    def run_write(path, name):
        gallery_box.image(path)
        gallery_box.write(name)
    def prog_writer(*args, **kwargs):
        return stqdm(*args, **kwargs, st_container=prog_box)
    image_writer = ImageWriter(5, image_box.image)
    title, runs = parse_yaml(state['yaml'])
    batch = BatchRunner(
        title,
        runs,
        run_args,
        image_writer=image_writer.write,
        run_writer=run_write,
        status_writer=top_status.write,
        tqdm=prog_writer
    )
    gallery = batch.run()
    zip_path = shutil.make_archive(title, format='zip', root_dir=gallery)
    top_status.write(zip_path)
    state['running'] = False
'''
Welcome to Artbot! Enter a prompt to get started. The image size is tuned for Colab, but the other settings can be changes as you wish.
`--` seperated prompts to switch midway through a run. By default it'll spend equal time on each prompt, but you can specify a ratio with `__`.
For example, `river--lava__2` will do half iterations on river and half on lava. The earlier iterations are more impactful, so a 1:1 ratio will skew to the earlier prompts.

You can also run prompts concurrently with `||`. This can be combined with the switching prompts to make complex prompts:
`river--lava__2||ocean waves`

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
- `fire lava--mountain water__1.2--ocean waves__1.4`
- `sunrise sunset horizon--ocean__2--forest__3`
- `mountains--bright sky||multiple--dynamic__2--frothy__3||ocean waves trending on artstation`

You can also add artist styles using `by` or `in the style of`, for example `Dynamic by Van Gogh`. The more art on wikiart by that artist the stronger their style.
Here's some of my favorites:
- `James Gurney`
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