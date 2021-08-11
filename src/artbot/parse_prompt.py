import re
from functools import reduce
from math import floor
# Structure is parts seperated by modifiers, starting and ending with a part:
#   part->mod->part->mod->part
# Parts are one or more concurrent prompts with an optional duration
# These will be added and removed over the course of the run
# Modifiers adjust the current prompt list somehow before adding the next one:
# --: Remove the newest prompt
# __: Remove the oldest prompt
# ++: Add the next prompt
# ||: Clear all prompts
# {n}: Run part n times longer than the other, can be placed anywhere in the part
# eg: 'a++b{2}--c__d||e&&f'
#   Run a -> add b -> remove b, add c -> remove a, add d -> remove everything, add e and f
#   b will be run twice as long as the others, i.e. 1:2:1:1 ratio
# Inside each part you can run prompts concurrently with &&:
# a&&b: Run a and b at the same time, they are treated as a pair for {} but not -- and __
#   This is because they're linked in the pst-, but once they're in the prompt list the link goes away
# You can reference other prompts and image outputs with '*':
# (image refs this aren't implemented in this file, but the token is defined here)
# prompt, image_prompt, init_image: '*prior_run||*predefined_prompt'
# predefined prompts are fields in the object root that only have a string
# { title: 'Woah', neon_prompt: 'Billowing Neon Lights', neon_run: { prompt: '*neon_prompt' } }

ref_tok = '*'
ratio_reg = r'{(.+)}' # we just cast the inside as a float
and_tok = '&&'
# a{2} => (2, [a])
# a&&b{2} {2}a&&b, a&{2}&b => (2, [a, b])
def parse_section(section):
    ratio = re.search(ratio_reg, section)
    ratio = float(ratio.groups()[0]) if ratio else 1
    prompt = re.sub(ratio_reg, '', section)
    return ratio, prompt.split(and_tok)

pop_new_tok = '--'
pop_old_tok = '__'
add_tok = '++'
clear_tok = '||'
# apply section modifier rules from above
# --, a&&b, [d, e] => {1: [d, a, b]}
def apply_mod(mod, part, old_prompts):
    new_prompts = old_prompts.copy()
    if mod == pop_new_tok:
        new_prompts.pop()
    if mod == pop_old_tok:
        new_prompts.pop(0)
    if mod == clear_tok:
        return part
    return part[0], new_prompts + part[1]

# take parts, list of prompts and list of mods
# mod and prompt list must be equal lengths
# return if there's no sections left
# else make new prompts with modification and recur
def apply_mods(parts, prompts_list, mods):
    if len(parts) == 0:
        return prompts_list
    part = parts.pop(0)
    mod = mods.pop(0)
    last_prompt = prompts_list[-1][1]
    mod_prompt = apply_mod(mod, part, last_prompt)
    prompts_list.append(mod_prompt)
    prompts_list = apply_mods(parts, prompts_list, mods)
    return prompts_list

# split into sections and modifiers
# sections contain '&&' and '{n}'(they aren't parsed here)
# a++b{2}--c||d&&e => ([a, b{2}, c, d&&e], [++, --, ||])
def get_parts(instr):
    mod_list = [pop_new_tok, pop_old_tok, add_tok, clear_tok]
    mod_reg = '|'.join(list(map(re.escape, mod_list)))
    sections = re.split(mod_reg, instr)
    mods = re.findall(f'({mod_reg})', instr)
    return sections, mods

# take a list of ratio, prompts and total iterations
# return a index of iteration => prompts beginning at that index
# [(1, [a]), (2, [a, b]), (1, [a, c]), (1, [d, e])], 500
# => {0: [a], 100: [a, b], 300: [a, c], 400: [d, e]}
def prompts_to_index(prompts_list, iterations):
    ratio_sum = reduce(lambda s, tup: s + tup[0], prompts_list, 0)
    prompts_dict = {}
    run_i = 0
    for prompt_ratio, prompt in prompts_list:
        prompts_dict[floor(run_i)] = prompt
        run_i += iterations * (prompt_ratio/ratio_sum)
    return prompts_dict

# take a prompt string and iterations, return a index of prompt changes
#def parse_prompt_string(string, iterations):
def parse_prompt(prompt, iterations):
    parts, mods = get_parts(prompt)
    parts = list(map(parse_section, parts)) # these are (ratio, [prompts])
    start_part = parts.pop(0)
    prompts_list = apply_mods(parts, [start_part], mods)
    return prompts_to_index(prompts_list, iterations)

# print(parse_prompt('a++b{2}--c__d||e&&f', 600))