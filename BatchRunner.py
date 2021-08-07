import threading
import math, time

# Run prompts concurrently
# For each core, try to run the top of the run list
# If it has unfulfilled requirements try the next one
# don't change the order, assume user ordered it as good as possible
class BatchRunner():
    def __init__(self, title, runs, runner):
        self.title = title
        # runs starts as an dict of args
        # as runs complete the output image is stored and run is removed
        self.runs = runs
        self.runner = runner
        self._lock = threading.Lock()
        self.outputs = {}
        self.gallery = f'Gaillery/{self.title}'
    
    # replace prompt chaining '*run' with output path, if it exists yet
    # If the output doesn't exist, return false
    def set_outputs(self, run):
        def parse(string):
            if '*' in string:
                in_run = string[1:]
                if in_run in self.outputs:
                    return self.outputs[in_run]
            return False
        init_image = parse(run['init_image'])
        image_prompts = map(parse, run['image_prompts'])
        if init_image and all(image_prompts):
            run['init_image'] = init_image
            run['image_prompts'] = image_prompts
            return run
        return False

    # iterate through the unfinished runs and do the next possible one
    # once that's done, save the output paths an remove it from the runlist
    def run_next(self):
        for run_name in self.runs:
            run = self.runs[run_name]
            parsed_run = self.set_outputs(run)
            if parsed_run:
                out_folder = f'{self.gallery}/{run_name}'
                def image_name_fn(iteration):
                    return f'{out_folder}/{iteration}-{math.floor(time.time())}.jpg'
                out_paths = self.runner(parsed_run, image_name_fn)
                self.outputs[run] = out_paths
                del self.runs[run]

    def run(self):
        while any(self.runs):
            self.run_next()
        return self.outputs