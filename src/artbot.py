#!/bin/python
import threading, os, shutil, ffpb
from sys import argv
from re import search
from torch.cuda import device_count
from runner import run_args
from parse import parse_yaml, update_ref, update_refs, ref_reg
from tqdm import tqdm
from output import write_video

def run_yaml(yaml, out_dir, image_writer, status_writer, tqdm)
    status_writer(False)
    settings, prompts, image_prompts = parse_yaml(yaml)
    print(f'Running "{settings['title']}" on device {dev}, saving output at {output}')
    outputs = run_prompts(settings, prompts, image_prompts, out_dir, image_writer=image_writer, status_writer=status_writer, tqdm=tqdm)
    if 'video' in settings.keys() and settings['video']:
        write_video(out_dir, name, outputs, settings['fps'], tqdm=self.tqdm)
    return outputs[-1]
