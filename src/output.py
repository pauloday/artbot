import os, shutil
import ffpb
from tqdm import tqdm
from json import dumps
from hashlib import md5
from pathlib import Path
from glob import glob

def str_hash(string):
    dhash = md5()
    dhash.update(string)
    return dhash.hexdigest()

def obj_hash(obj):
    encoded = dumps(obj, sort_keys=True).encode()
    return str_hash(encoded)

def image_name(out_dir, i, run):
    image_name = f'{out_dir}/{i}_{run["size"][0]}x{run["size"][1]}.jpg'
    dump_args = run.copy();
    del dump_args['iterations'] # only arg that doesn't effect each image
    return output_file_postfix(image_name, obj_hash(dump_args))

# output a file path with a hash of something included, and create any parent dirs
def output_file_postfix(path, postfix):
    path = Path(path)
    parent = path.parent.absolute()
    if not os.path.exists(parent):
            os.makedirs(parent)
    fil, ext = os.path.splitext(path)
    outpath = f'{fil}-{postfix}{ext}'
    return outpath

# returns (dir, should do run)
# if should do run, dir is output, else dir is output image
def get_next_path(name, gallery, run):
    out_dir = f'{gallery}/{name}'
    out_path = image_name(out_dir, run['iterations'], run)
    if os.path.exists(out_path):
        return out_path, False
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir, True

# function to write a video to a file
# use the str hash function
# with the list of filenames (dir not included) as hash
def write_video(out_dir, name, outputs, tqdm=tqdm):
    tmp_dir = f'{out_dir}/tmp'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    i = 0
    for frame in outputs:
        f = f'{tmp_dir}/{str(i).zfill(5)}.jpg'
        shutil.copyfile(frame, f)
        i += 1
    video_file = f'{out_dir}/{name}.mp4'
    vid_path = output_file_postfix(video_file, obj_hash(outputs))
    argv = ['-r', '24', '-f', 'image2', '-i', f'{tmp_dir}/%05d.jpg', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', vid_path]
    ffpb.main(argv, tqdm=tqdm)
    shutil.rmtree(tmp_dir)
    return vid_path