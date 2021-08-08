from parse import parse_yaml
from BatchRunner import BatchRunner
from runner import run_args
import torch
import shutil
from os.path import basename

def dev_count():
    return torch.cuda.device_count()

# takes an input yaml file and runs it then returns a zip of the results
def artbot(infile):
    title, runs = parse_yaml(open(infile).read())
    batch = BatchRunner(title, runs, run_args)
    gallery = batch.run()
    shutil.copyfile(infile, f'{gallery}/{basename(infile)}')
    return shutil.make_archive(title, format='zip', root_dir=gallery)