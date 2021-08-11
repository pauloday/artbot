#!/bin/python
import threading, os, shutil, ffpb
from sys import argv, path
from torch.cuda import device_count
from runner import run_args, get_image_name
from parse import parse_yaml
from tqdm import tqdm
from glob import glob
path.append('./taming-transformers')

# this should only be run as a script
def is_runnable(run):
    params = [run['init_image'], run['image_prompts']]
    return all(map(lambda r: not (r and '*' in r), params))

# check if this run (size and iterations) was done already
def has_output(run, out_folder):
    checkpoint = glob(f'{out_folder}/{run["iterations"]}*.jpg')
    return len(checkpoint) != 0 and checkpoint[0]

def update_ref(ref, name, path):
    def update_one_ref(r):
        return path if r and r[1:] == name else r
    def update_dict(d):
        return {i: update_one_ref(p) for i, p in d.items()}
    if type(ref) == list: # each prompt may be swap: {0: a, 50: b}
        return list(map(
            lambda r: update_dict(r) if type(r) == dict else update_one_ref(r),
            ref
        ))
    return update_one_ref(ref)

# returns (dir, should do run)
# if should do run, dir is output, else dir is output image
def get_next_path(name, gallery, run):
    out_dir = f'{gallery}/{name}'
    out_path = get_image_name(out_dir, run['iterations'], run)
    if os.path.exists(out_path):
        return out_path, False
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir, True

# keep track of which runs we can do in a thread safe way
class RunGetter():
    def __init__(self, runs):
        self.runs = runs
        self._lock = threading.Lock()
    # update *refs with an actual path
    def update_runs(self, name, path):
        with self._lock:
            for run in self.runs.values():
                run['init_image'] = update_ref(run['init_image'], name, path)
                run['image_prompts'] = update_ref(run['image_prompts'], name, path)
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
    def __init__(self, yaml, image_writer=False):
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
        outputs = run_args(run, output, dev=dev, image_writer=self.image_writer)
        self.index.toggle(dev, True)
        self.__finish_run(name, outputs)
        return outputs[-1]

    #TODO: video generation elsewhere, keep Artbot small
    def __finish_run(self, name, outputs):
        tmp_dir = f'{self.gallery}/tmp'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        i = 0
        for frame in outputs:
            if os.path.exists(frame):
                shutil.copyfile(frame, f'{tmp_dir}/{str(i).zfill(4)}.jpg')
                i += 1
        video_name = f'{self.gallery}/{name}.mp4'
        argv = ['-r', '24', '-f', 'image2', '-i', f'{tmp_dir}/%04d.jpg', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', video_name]
        ffpb.main(argv, tqdm=tqdm)
        shutil.rmtree(tmp_dir)
        if self.image_writer:
                self.image_writer(video_name, video=True)
        return video_name

if __name__ == "__main__":
    in_dict = parse_yaml(open(argv[0]).read())
    bot = Artbot(in_dict)
    bot.run()