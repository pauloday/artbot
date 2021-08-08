import os, threading
import torch
import math, time
import shutil
import ffpb
from tqdm import tqdm as default_tqdm
from glob import glob

# Run prompts concurrently
# For each core, try to run the top of the run list
# If it has unfulfilled requirements try the next one
# don't change the order, assume user ordered it as good as possible
class BatchRunner():
    def __init__(self, title, runs, runner, image_writer=False, run_writer=False, tqdm=default_tqdm, status_writer=False):
        self.title = title
        # runs starts as an dict of args
        # as runs complete the output image is stored in place of run
        self.runs = runs
        self.runner = runner
        self._lock = threading.Lock()
        self.image_writer = image_writer
        self.run_writer = run_writer
        self.status_writer = status_writer
        self.tqdm = tqdm
        self.gallery = f'Gaillery/{self.title}'
        if not os.path.exists(self.gallery):
            os.makedirs(self.gallery)
    
    # replace prompt chaining '*run' with output path, if it exists yet
    # If the output doesn't exist, return false
    #TODO: break this up
    def set_outputs(self, run):
        def get_ref(string):
            if string and '*' in string:
                    in_run = string[1:]
                    if type(self.runs[in_run]) == list:
                        return self.runs[in_run][-1]
                    else:
                        return False
        def is_ref(string):
            return string and type(string) == str and '*' in string
        init_image = run['init_image']
        image_prompts = run['image_prompts']
        image_prompts = image_prompts if image_prompts != None else []
        prompt_is_ref = all(list(map(is_ref, image_prompts))) if image_prompts != [] else False
        init_is_ref = is_ref(init_image)
        if not init_is_ref and not prompt_is_ref: # assume no refs means this is ready
            return run
        ref_prompt = list(map(get_ref, image_prompts))
        ref_init = get_ref(init_image)
        if (prompt_is_ref and not all(ref_prompt)) or (init_is_ref and not ref_init):
            return False # one of the refs isn't fetchable yet, no point fetching the other
        if init_is_ref and ref_init:
            run['init_image'] = ref_init
            ret_run = run
        if prompt_is_ref and ref_prompt:
            run['image_prompts'] = ref_prompt
            ret_run = run
        return ret_run
        
    def make_video(self, title, frames, video_name):
        tmp_dir = f'{self.gallery}/tmp'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        i = 0
        for frame in frames:
            if os.path.exists(frame):
                shutil.copyfile(frame, f'{tmp_dir}/{str(i).zfill(4)}.jpg')
                i += 1
        argv = ['-r', '20', '-f', 'image2', '-i', f'{tmp_dir}/%04d.jpg', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_name]
        ffpb.main(argv, tqdm=self.tqdm)
        shutil.rmtree(tmp_dir)

    # iterate through the unfinished runs and do the next possible one
    # once that's done, save the output paths in place of the run args
    # theoretically this can just be run once per core an it works (with a lock)
    # TODO: this is a big boy, cut it down
    def run_next(self):
        for run_name in self.runs:
            run = self.runs[run_name]
            if type(run) == dict: # run args is dict - eventually make it class
                parsed_run = self.set_outputs(run)
                if parsed_run: # this run is ready
                    if self.status_writer:
                        self.status_writer(f'{run_name}:')
                    i_format = parsed_run['format'] if 'format' in parsed_run else 'jpg'
                    out_folder = f'{self.gallery}/{run_name}'
                    if not os.path.exists(out_folder):
                        os.makedirs(out_folder)
                    #TODO: passing this into run is probably slowing it down a lot
                    # since it has to keep a closure of all the stuff in here
                    # ideally we pre make the names and pass them in as strings
                    def image_name_fn(name):
                        return f'{out_folder}/{name}-{math.floor(time.time())}.{i_format}'
                    # check to see if the output of this run exists
                    # if it does, skip the run and give the user a message
                    checkpoint = glob(f'{out_folder}/{parsed_run["iterations"]}*.{i_format}')
                    final_out = f'{self.gallery}/{run_name}.mp4'
                    if len(checkpoint) != 0:
                        print(f'Found output for {run_name} at {checkpoint[0]}, skipping run')
                        self.runs[run_name] = checkpoint
                        final_out = checkpoint
                    else:
                        print(f'\nDoing run "{run_name}". Saving output in {out_folder}')
                        out_paths = self.runner(parsed_run, image_name_fn, dev=0, image_writer=self.image_writer, tqdm=self.tqdm)
                        if 'video' in run and run['video']:
                            self.make_video(run_name, out_paths)
                        else: # no video written to gallery, so put the last output there instead
                            out_folder = self.gallery
                            final_out = image_name_fn(run_name)
                            shutil.copyfile(out_paths[-1], final_out)
                        self.runs[run_name] = out_paths
                    if self.run_writer:
                        self.run_writer(final_out, run_name)
                    torch.cuda.empty_cache()
                    self.run_next()

    def run(self):
        self.run_next()
        return self.gallery