import util
import os
import pprint
import math
import time
import runner

# Get the next prompt in a multithread safe way (avoids a race condition)
# get_next returns either the next prompt or False if there are none left
class PromptGetter():
    def __init__(self, prompts):
        self.prompts = prompts
        self.index = -1 # start at -1 so we can always add 1 before fetching indexed val
        self._lock = threading.Lock()
    def get_next(self):
        with self._lock:
            if self.index == len(self.prompts) - 1:
                return False
            self.index += 1
            return self.prompts[self.index]

# defines a single run
class Single:
    def __init__(self, args, title=''):
        self.title = title if title != '' else args['prompts']
        self.prompt = args['prompts']
        self.image_prompts = args['image_prompts']
        self.args = args
        # set up the gallery folders
        gallery = util.windows_path_sanitize(f'/content/Gaillery/{title}')
        self.args['gallery'] = gallery
        self.image_prompts_folder = f'{gallery}/image_prompts'
        self.info_folder = f'{gallery}/info'
        self.info_string = f'prompts = {self.prompt}\nimage_prompts = {self.image_prompts}'
        if not os.path.exists(gallery):
            os.makedirs(gallery)
        if not os.path.exists(self.image_prompts_folder):
            os.makedirs(self.image_prompts_folder)
        if not os.path.exists(self.info_folder):
            os.makedirs(self.info_folder)
        
    def write_info(self):
        info_string = f'{self}\n{self.info_string}\n{pprint.pformat(self.args)}'
        # write with timestamp to preserve info from reruns
        info_path = f'{self.info_folder}/info-{math.floor(time.time())}.txt'
        file = open(info_path, 'w+')
        file.write(info_string)
        file.close()
        print(f'Wrote {file.name}')

    def run(self, update_box, add_to_video, dev=0, image_name=None):
        #if runner is passed a string it will do each letter as an individaul prompt
        runner.run_prompt(
            self.args,
            update_box,
            add_to_video,
            dev,
            image_name
        )
  