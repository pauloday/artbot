import sys
import math
import io
from pathlib import Path
import os
import concurrent.futures
import util

from IPython import display
from omegaconf import OmegaConf
from PIL import Image
import requests
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from CLIP import clip
import shutil
import threading
import itertools
import time
from stqdm import stqdm


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


replace_grad = ReplaceGrad.apply


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


clamp_with_grad = ClampWithGrad.apply


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 3)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)

# parses a schedule into a single prompt
# schedule example:
# [('tree', 1), ('river', 1)]
# tree 50% of the time, river 50%
# so if we're below 5/10 return tree, 5/10 or above return river
# i_pct is % of way through iterations
def get_current_prompt(schedule, i_pct):
    sched_pct = 0
    ratio_sum = 0
    for prompt in schedule:
        ratio_sum += prompt[1]
    for prompt in schedule:
        sched_pct += prompt[1]/ratio_sum
        if i_pct < sched_pct:
            return prompt[0]
    return schedule[-1][0]

# rate as in only display a image every n seconds
class ImageWriter():
    def __init__(self, rate, writer):
        self.rate = rate
        self.out_stamp = math.floor(time.time())
        self.writer = writer
    
    def write(self, image):
        now = math.floor(time.time())
        if now - self.out_stamp > self.rate:
            self.out_stamp = now
            self.writer(image)


# prompts here is a single image's prompts, not a batch of prompts
# each prompt can be an array of tuples with ('prompt', ratio)
# ratio is the time spent on the prompt relative to the others in the array
# so [('space', 1), ('ocean', 1)] will do space for 50% iterations, then ocean
def run_prompt(args, update_box, add_frame, dev=0, image_name=None,):
    image_box = update_box.empty()
    image_writer = ImageWriter(5, image_box.image) # hand tuned to never clobber the output with ngrok free tier
    bottom_status = update_box.empty()
    device_name = f'cuda:{dev}'
    device = torch.device(device_name)
    print('Using device:', device, args['vqgan_checkpoint'])

    model = load_vqgan_model(args['vqgan_config'], args['vqgan_checkpoint']).to(device)
    perceptor = clip.load(args['clip_model'], jit=False)[0].eval().requires_grad_(False).to(device)

    cut_size = perceptor.visual.input_resolution
    e_dim = model.quantize.e_dim
    f = 2**(model.decoder.num_resolutions - 1)
    make_cutouts = MakeCutouts(cut_size, args['cutn'], cut_pow=args['cut_pow'])
    n_toks = model.quantize.n_e
    toksX, toksY = args['size'][0] // f, args['size'][1] // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    if args['seed'] is not None:
        torch.manual_seed(args['seed'])

    if args['init_image'] is not None:
        pil_image = Image.open(fetch(args['init_image'])).convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    z_orig = z.clone()
    z.requires_grad_(True)
    opt = optim.Adam([z], lr=args['step_size'])

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])

    pMs = []

    def set_prompts(i):
        for p in args['prompts']:
            p_str = p[0] if type(p) == tuple else p
            prompt = get_current_prompt(p, i/args['iterations']) if type(p) == list else p_str
            if prompt != '_': # allow empty prompts for placeholder
                curr_ratio_prompt = prompt
                txt, weight, stop = parse_prompt(prompt)
                embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
                pMs.append(Prompt(embed, weight, stop).to(device))

        if args['image_prompts'] is not None:
            for prompt in args['image_prompts']:
                path, weight, stop = parse_prompt(prompt)
                img = resize_image(Image.open(fetch(path)).convert('RGB'), (sideX, sideY))
                batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
                embed = perceptor.encode_image(normalize(batch)).float()
                pMs.append(Prompt(embed, weight, stop).to(device))

        for seed, weight in zip(args['noise_prompt_seeds'], args['noise_prompt_weights']):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
            pMs.append(Prompt(embed, weight).to(device))
    set_prompts(0)
    
    def synth(z):
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    def checkin(i, losses, out_path):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        # this is some math thing I don't want to get rid of, but it takes a lot of space for bigger runs
        #print(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        out = synth(z)
        TF.to_pil_image(out[0].cpu()).save(out_path)
        image_writer.write(out_path)
        add_frame(out_path)
        bottom_status.write(f'Wrote {out_path}')

    def ascend_txt():
        out = synth(z)
        iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

        result = []

        if args['init_weight']:
            result.append(F.mse_loss(z, z_orig) * args['init_weight'] / 2)

        for prompt in pMs:
            result.append(prompt(iii))

        return result

    def train(i):
        new_stamp = limit_stamp
        opt.zero_grad()
        lossAll = ascend_txt()
        display_freq = math.floor(args['iterations']/args['images_per_prompt'])
        out_path = util.image_path(args, i)
        if image_name:
            out_path = image_name(args['prompts'], i)
        if (i % display_freq == 0 and i != 0):
            checkin(i, lossAll, out_path)
        loss = sum(lossAll)
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.copy_(z.maximum(z_min).minimum(z_max))
        return out_path

    i = 0
    try:
        with stqdm(total=args['iterations'] + 1, st_container=update_box) as pbar:
            limit_stamp = math.floor(time.time())
            displayed_images = 0
            last_image = False
            while i <= args['iterations']:
                last_image = train(i)
                set_prompts(i)
                i += 1
                pbar.update()
    except KeyboardInterrupt:
        pass
