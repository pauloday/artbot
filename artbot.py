from parse import parse_yaml
from BatchRunner import BatchRunner
from runner import run_args
import torch
import shutil
from os.path import basename

def dev_count():
    return torch.cuda.device_count()

# takes an input yaml file and runs it then returns a zip of the results
def artbot(instr):
    title, runs = parse_yaml(instr)
    batch = BatchRunner(title, runs, run_args)
    gallery = batch.run()
    open(f'{gallery}/{title}.yml', 'w').write(str(instr))
    return shutil.make_archive(title, format='zip', root_dir=gallery)