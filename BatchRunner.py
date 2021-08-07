import os, threading
import math, time

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
        print(run)
        def parse(string):
            if string and string != None and type(string) == str:
                if '*' in string:
                    in_run = string[1:]
                    if type(self.runs[in_run]) == list:
                        return self.runs[in_run[-1]]
                else:
                    return string # valid string but not a pointer
            return False # not a valid string, or a string with '*' that can't be replaced
        image_prompts = run['image_prompts']
        init_image = parse(run['init_image'])
        image_prompts = list(map(parse, image_prompts if image_prompts != None else []))
        if init_image:
            run['init_image'] = init_image
        if all(image_prompts):
            run['image_prompts'] = image_prompts
        if init_image and all(image_prompts):
            return run
        else:
            return False

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
                    out_paths = self.runner(parsed_run, image_name_fn)
                    self.runs[run_name] = out_paths
                    self.run_next()

    def run(self):
        self.run_next()
        return self.runs