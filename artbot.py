from parse import parse_yaml
from BatchRunner import BatchRunner
from runner import run_args
import torch
import shutil

def dev_count():
    return torch.cuda.device_count()

# takes an input yaml str and runs it then returns a zip of the results
def artbot(instr):
    title, runs = parse_yaml(instr)
    batch = BatchRunner(title, runs, run_args)
    gallery = batch.run()
    return shutil.make_archive(gallery, format='zip', root_dir='Gaillery')