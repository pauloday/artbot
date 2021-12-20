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

def image_name(out_dir, i, settings):
    image_name = f'{out_dir}/{i}_{settings["size"][0]}x{settings["size"][1]}.jpg'
    dump_args = run.copy();
    # these args shouldn't ever change individual images when modified
    del dump_args['iterations']
    del dump_args['video']
    return output_file_postfix(image_name, obj_hash(dump_args))

# function to write a video to a file
# use the str hash function
# with the list of filenames (dir not included) as hash
def write_video(out_dir, name, outputs, fps, tqdm=tqdm):
    tmp_dir = f'{out_dir}/tmp-{name}'
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
    argv = ['-r', str(fps), '-f', 'image2', '-i', f'{tmp_dir}/%05d.jpg', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', vid_path]
    ffpb.main(argv, tqdm=tqdm)
    shutil.rmtree(tmp_dir)
    return vid_path