from parse import parse_yaml
from BatchRunner import BatchRunner
import torch

def dev_count():
    return torch.cuda.device_count()

# takes an input yaml str and runs it
def artbot(instr):
    runs = parse_yaml(instr)
    batch = BatchRunner(runs)
    print(batch.run())