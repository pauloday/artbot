from parse import parse_yaml
from BatchRunner import BatchRunner
from runner import run_args
import torch

def dev_count():
    return torch.cuda.device_count()

# takes an input yaml str and runs it
def artbot(instr):
    title, runs = parse_yaml(instr)
    batch = BatchRunner(title, runs, run_args)
    batch.run()


artbot(open('template.yml').read())