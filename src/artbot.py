#!/bin/python
import shutil
from runner import run_prompts
from parse import parse_yaml, flatten_array, expand_iteration
from tqdm import tqdm
from output import write_video

def run_yaml(yaml, out_dir, image_writer, status_writer, tqdm):
    status_writer(False, 0)
    settings, prompts, image_prompts = parse_yaml(yaml)
    outputs = run_prompts(settings, prompts, image_prompts, out_dir, image_writer=image_writer, status_writer=status_writer, tqdm=tqdm)
    if 'video' in settings.keys() and settings['video']:
        write_video(out_dir, name, outputs, settings['fps'], tqdm=self.tqdm)
    shutil.copy(outputs[-1], out_dir)

def run_array(settings, prompts, image_prompts, out_dir, image_writer, status_writer, tqdm):
    status_writer(False, 0)
    parsed_prompts = flatten_array(map(expand_iteration, prompts))
    outputs = run_prompts(settings, parsed_prompts, image_prompts, out_dir, image_writer=image_writer, status_writer=status_writer, tqdm=tqdm)
    if 'video' in settings.keys() and settings['video']:
        write_video(out_dir, name, outputs, settings['fps'], tqdm=self.tqdm)
    shutil.copy(outputs[-1], out_dir)