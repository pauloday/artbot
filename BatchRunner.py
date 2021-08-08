import os, threading
import torch
import math, time
import shutil
import ffpb
import tqdm
from glob import glob

# Run prompts concurrently
# For each core, try to run the top of the run list
# If it has unfulfilled requirements try the next one
# don't change the order, assume user ordered it as good as possible
class BatchRunner():
    def __init__(self, title, runs, runner):
        self.title = title
        # runs starts as an dict of args
        # as runs complete the output image is stored in place of run
        self.runs = runs
        self.runner = runner
        self._lock = threading.Lock()
        self.gallery = f'Gaillery/{self.title}'
        if not os.path.exists(self.gallery):
            os.makedirs(self.gallery)
    
    # replace prompt chaining '*run' with output path, if it exists yet
    # If the output doesn't exist, return false
    def set_outputs(self, run):
        # get the output of a prior run
        # if the run hasn't finished, return false
        def get_ref(string):
            if string and '*' in string:
                    in_run = string[1:]
                    if type(self.runs[in_run]) == list:
                        return self.runs[in_run][-1]
                    else:
                        return False
        # check a prompt to see if it can be run yet
        # if it's a ref ('*run'), and it can be fetched, return True
        # if it's not fetchable, return false
        # otherwise it's not a ref so return true
        def is_ref(string):
            return string and type(string) == str and '*' in string
        init_image = run['init_image']
        image_prompts = run['image_prompts']
        image_prompts = image_prompts if image_prompts != None else []
        prompt_is_ref = all(list(map(is_ref, image_prompts))) if image_prompts != [] else False
        init_is_ref = is_ref(init_image)
        if not init_is_ref and not prompt_is_ref:
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
        
    def make_video(self, frames):
        tmp_dir = f'{self.gallery}/tmp'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        i = 0
        for frame in frames:
            print(frame)
            if os.path.exists(frame):
                shutil.copyfile(frame, f'{tmp_dir}/{str(i).zfill(4)}.jpg')
            i += 1
        video_name = f'{self.gallery}/{math.floor(time.time())}.mp4'

        argv = ['-r', '20', '-f', 'image2', '-i', f'{tmp_dir}/%04d.jpg', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_name]
        ffpb.main(argv, tqdm=tqdm)
        shutil.rmtree(tmp_dir)

    # iterate through the unfinished runs and do the next possible one
    # once that's done, save the output paths in place of the run args
    # theoretically this can just be run once per core an it works (with a lock)
    def run_next(self):
        for run_name in self.runs:
            run = self.runs[run_name]
            if type(run) == dict: # run args is dict - eventually make it class
                parsed_run = self.set_outputs(run)
                if parsed_run: # this run is ready
                    out_folder = f'{self.gallery}/{run_name}'
                    if not os.path.exists(out_folder):
                        os.makedirs(out_folder)
                    def image_name_fn(iteration):
                        return f'{out_folder}/{iteration}-{math.floor(time.time())}.jpg'
                    # check to see if the output of this run (i.e. {iteration}-.*.jpg) exists
                    # if it does, skip the run and give the user a message
                    checkpoint = glob(f'{out_folder}/{parsed_run["iterations"]}*.jpg')
                    if len(checkpoint) != 0:
                        print(f'Found previous output at {checkpoint[0]}, skipping run')
                        self.runs[run_name] = checkpoint
                    else:
                        print(f'Doing run "{run_name}". Saving output in {out_folder}')
                        out_paths = self.runner(parsed_run, image_name_fn, dev=0)
                        self.make_video(out_paths)
                        self.runs[run_name] = out_paths
                    torch.cuda.empty_cache()
                    self.run_next()

    def run(self):
        self.run_next()
        return self.gallery