#import runner as r
import util as u

def dev_count():
    return torch.cuda.device_count()
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
class Run:
    def __init__(self, prompt, image_prompts, args, title=None):
        self.title = title if title else prompt
        self.prompt = prompt
        self.image_prompts = image_prompts
        self.args = args
        # set up the gallery folders
        self.args.gallery = u.windows_path_sanitize(f'Gaillery/{title}')
        self.args.image_prompts_folder = f'{args.gallery}/image_prompts'
        self.info_folder = f'{self.args.gallery}/info'
        self.info_string = f'prompts = {self.prompt}\nimage_prompts = {image_prompts}'
        if not os.path.exists(self.args.gallery):
            os.makedirs(self.args.gallery)
        if not os.path.exists(self.args.image_prompts_folder):
            os.makedirs(self.args.image_prompts_folder)
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
        
    # def run(self, dev=0, image_name=None):
    #     #if runner is passed a string it will do each letter as an individaul prompt
    #     runner.run_prompt(
    #         self.prompt if type(self.prompt) == list else [self.prompt],
    #         self.image_prompts,
    #         self.args, 
    #         dev,
    #         image_name
    #     )
        
# the most basic one is a list of prompts
class Batch(Run):
    def __init__(self, prompts, image_prompts, args, title=None):
        self.title = title if title else '-'.join(prompts)
        # prompt gets set in run, so pass nothing initially
        Run.__init__(self, 'init', image_prompts, args, title)
        self.getter = PromptGetter(prompts)
        # this is for logging, the info specific to this batch type
        self.info_string = f'prompts = {self.getter.prompts}\nimage_prompts = {image_prompts}'
        
    def run(self, image_name=None):
        while self.prompt:
            threads = list()
            for dev in range(r.dev_count()):
                self.prompt = self.getter.get_next()
                if self.prompt:
                    thread = threading.Thread(target=Run.run, args=(self, dev, image_name))
                    threads.append(thread)
                    thread.start()
            for index, thread in enumerate(threads):
                thread.join()


# This one takes a base and a list of postfixes, then combines them
class PostBatch(Batch):
    def __init__(self, base, postfixes, image_prompts, args):
        prompts = list(map(lambda post: f'{base} {post}', postfixes))
        Batch.__init__(self, prompts, image_prompts, args)
        self.info_string = f'base = {self.base}\npostfixes = {self.postfixes}'
        

# Take a base and a list of postfix combo pieces
# Then make a prompt for base + each combo of the pieces
# e.g. 'p' [1, 2] -> 'p', 'p 1', 'p 2', 'p 1 2'
class ComboBatch(PostBatch):
    def __init__(self, base, pieces, image_prompts, args, joiner=' ; '):
        postfixes = []
        join = lambda c: joiner.join(c)
        for n in range(len(pieces) + 1):
            combos = itertools.combinations(pieces, n)
            posts = list(map(join, combos))
            postfixes += posts
        PostBatch.__init__(self, base, postfixes, image_prompts, args)
        self.info_string = f'base = {self.base}\npieces = {self.pieces}'


# 1. do i iterations for the first prompt
# 2. take that output as the starting image for the next prompt
# progress through all prompts this way until out of prompts
class ProgBatch(Run):
    def __init__(self, prompts, image_prompts, args, cycles=1):
        self.all_prompts = prompts
        # to ensure we don't overwrite
        self.image_name_counter = 0
        self._lock = threading.Lock()
        self.cycles = cycles
        # the prompt here is only used for the gallery name
        title = map(lambda p: p[0] if type(p) == tuple else p, prompts)
        Run.__init__(self, '-into-'.join(title), image_prompts, args)
    
    def image_name(self, prompt, i):
        return windows_path_sanitize(f'{self.args.gallery}/{self.image_name_counter}-{prompt}-{i}.jpg')
    
    def run(self):
        next_base = None
        args.images_per_prompt = 1
        for cycle in range(self.cycles):
            self.cycle = cycle
            for prompt in self.all_prompts:
                if type(prompt) == tuple:
                    self.prompt = prompt[0]
                    self.args.iterations = prompt[1]
                else:
                    self.prompt = prompt
                args.init_image = next_base
                Run.run(self, image_name=self.image_name)
                next_base = self.image_name(self.args.iterations)
                with self._lock:
                    self.image_name_counter += 1
                
    
# 0. only use n prompts, where n = the number of cores
# 1. do i iterations for each prompt concurrently (i.e. a batch)
# 2. cycle the output images into the starting image for the next code
# 3. repeat for c cycles
class BraidBatch(Batch):
    def __init__(self, prompts, image_prompts, args, cycles=dev_count):
        self.prompts = prompts
        # to ensure we don't overwrite
        self.image_name_counter = 0
        self._lock = threading.Lock()
        if len(self.prompts) > dev_count:
            self.prompts = prompts[0 : dev_count]
            print('WARNING: Truncating prompts to {self.all_prompts}')
        self.cycles = cycles
        Batch.__init__(self, self.prompts, image_prompts, args)

    def image_name(self, prompt, i):
        return windows_path_sanitize(f'{self.args.gallery}/{self.image_name_counter}-{prompt}-{i}.jpg')
    
    def run(self):
        for cycle in range(self.cycles):
            threads = list()
            for dev in range(dev_count):
                self.prompt = self.getter.get_next()
                if self.prompt:
                    thread = threading.Thread(target=Run.run, args=(self, dev, self.image_name))
                    threads.append(thread)
                    thread.start()
            for index, thread in enumerate(threads):
                thread.join()
            self.prompts = self.prompts[1:] + self.prompts[:1]
            self.getter = PromptGetter(self.prompts)
            self.image_name_counter += 1

# 1. For each prompt generate i iterations
# 2. Use those images as prompts along with all the prompts merged for another generation
# 3. Use that as an image prompt along with each individual prompt again
# 4. Repeat until we hit a set number of cycles (1 cycle = split -> merged)