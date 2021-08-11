#!/bin/python
import threading, os, shutil, ffpb
from sys import argv
from re import search
from torch.cuda import device_count
from runner import run_args, get_image_name
from parse import parse_yaml, update_ref, update_refs, ref_reg
from tqdm import tqdm
from glob import glob
from output import write_video, get_next_path

def is_runnable(run):
    params = [run['init_image'], run['image_prompt']]
    has_ref = lambda r: not (r and search(ref_reg, r))
    return all(map(has_ref, params))

# check if this run (size and iterations) was done already
def has_output(run, out_folder):
    checkpoint = glob(f'{out_folder}/{run["iterations"]}*.jpg')
    return len(checkpoint) != 0 and checkpoint[0]

# keep track of which runs we can do in a thread safe way
class RunGetter():
    def __init__(self, runs):
        self.runs = runs
        self._lock = threading.Lock()
    # update *refs with an actual path
    def update_runs(self, name, path):
        with self._lock:
            for run in self.runs.values():
                if run['init_image']:
                    run['init_image'] = update_ref(run['init_image'], name, path)
                if run['image_prompt']:
                    print(run['image_prompt'], name, path)
                    run['image_prompt'] = update_refs(run['image_prompt'], name, path)
    def get_next(self):
        with self._lock:
            for name in self.runs:
                if is_runnable(self.runs[name]):
                    return name, self.runs.pop(name)
        return False, False

# keep track of which devices are free in athread safe way
class DevIndex():
    def __init__(self, devs):
        self.devs = {d: True for d in range(devs)}
        self._lock = threading.Lock()
    def toggle(self, dev, state):
        with self._lock:
            self.devs[dev] = state
    # returns a list of numbers, the ready cores
    def get_ready(self):
        with self._lock:
            c = list(filter(lambda e: self.devs[e], self.devs))
            return c

class Artbot():
    def __init__(self, yaml, image_writer=False, tqdm=tqdm):
        title, runs = parse_yaml(yaml)
        self.getter = RunGetter(runs)
        self.index = DevIndex(device_count())
        self.gallery = f'Gaillery/{title}'
        self.image_writer = image_writer
        self.test = False
        if not os.path.exists(self.gallery):
            os.makedirs(self.gallery)
        with open(f'{self.gallery}/{title}.yml', 'wb') as f:
            f.write(yaml)

    def run(self):
        threads = list()
        for dev in self.index.get_ready():
            thread = threading.Thread(target=self.__run_dev, args=[dev])
            threads.append(thread)
            thread.start()
        for _, thread in enumerate(threads):
            thread.join()

    # recursive, does runs until it can't find ready cores/runs
    # once a run is done, it tries to start another one for all available cores
    def __run_dev(self, dev):
        run_name, run = self.getter.get_next()
        if run:
            output, should_run = get_next_path(run_name, self.gallery, run)
            if should_run:
                output = self.__do_run(run, run_name, output, dev)
            else:
                print(f'Finished run "{run_name}" output found at {output}, skipping')
            self.getter.update_runs(run_name, output)
            for d in self.index.get_ready():
                self.__run_dev(d)
            self.test = 'finished'

    def __do_run(self, run, name, output, dev):
        print(f'Running "{name}" on device {dev}, saving output at {output}')
        self.index.toggle(dev, False)
        outputs = run_args(run, output, dev=dev, image_writer=self.image_writer, tqdm=tqdm)
        self.index.toggle(dev, True)
        write_video(name, outputs)
        return outputs[-1]

if __name__ == "__main__":
    in_dict = parse_yaml(open(argv[0]).read())
    bot = Artbot(in_dict)
    bot.run()