
postcards = [
    [('sunrise sunset horizon', 1), ('ocean', 2), ('forest', 3), ('sunrise sunset horizon ocean forest in the style of Claude Monet ArtstationHQ', 4)],
    [('sunrise sunset horizon in the style of Studio Ghibli', 1),
     ('ocean in the style of Studio Ghibli', 2), ('forest in the style of Studio Ghibli', 3),
     ('sunrise sunset horizon ocean forest in the style of Claude Monet ArtstationHQ', 4)],
]

vast =  [[('sunrise sunset horizon in the style of Studio Ghibli', 1), ('ocean in the style of Studio Ghibli', 2), ('forest in the style of Studio Ghibli', 3),
    ('sunrise sunset horizon ocean forest in the style of Studio Ghibli ArtstationHQ', 4)],
    [('sunrise sunset horizon ArtstationHQ', 1), ('ocean ArtstationHQ', 2), ('forest ArtstationHQ', 3),
    ('sunrise sunset horizon ocean forest in the style of Studio Ghibli', 4)]]

vast2 = ['sunrise sunset sky cloud balloons in the style of Van Gogh', 'rocky mountain valley forest landscape in the style of Salvador Dali']
oppo = [
    [('fire lava', 1), ('mountain water', 1), ('ocean waves by Van Gogh ArtstationHQ', 1)],
    [('fire lava', 1), ('mountain water', 1), ('ocean waves by Van Gogh trending on Artstation', 1)],
    [('fire lava by trending on Artstation', 1), ('mountain water by Van Gogh', 1), ('ocean waves', 1)],
    [('fire lava by Van Gogh', 1), ('mountain water by Van Gogh', 1), ('ocean waves by trending on Artstation', 1)]
]
    
tmp2 = ['landscape in the style of Claude Monet', 'brutalist archetecture in the style of M.C. Escher']

upscale = [
    [[('fire lava', 1), ('mountain water', 1), ('ocean waves trending on Artstation', 1)]],
    [[('bald man', 1), ('rocket ship', 1), ('Jeff Bezos', 0.5), ('industrial hell trending on artstation', 0.5)]],
    [[('bright sky', 1), ('bridges', 1), ('lush valley by Claude Monet', 2)]],
    [[('bright sky by Van Gogh', 1), ('bridges by M.C Escher', 1), ('lush valley by Claude Monet', 2)]],
    [[('outer space', 1), ('ocean waves', 2), ('forest', 3), ('ArtstationHQ', 1)]],
    [('Terminator', 1),('dell', 1), ('China', 1)]
]


# list of cool styles, use with post/combo batches
render_prompts = [
    'artstation',
    'trending on artstation',
    'artstationHQ',
    'vray',
    'photograph',
    'abstract',
    'matte painting'
]

# same for 'in the style of x'
artist_prompts = list(map(lambda a: f'in the style of {a}',
                          ['Claude Monet',
                           'Van Gogh',
                           'Salvador Dali',
                           'Alex Grey',
                           'M.C. Escher',
                           'Studio Ghibli']))

twok_dim = [1820, 1026]
onesixnine_a100 = [1600, 900]
three2_dim = [1656, 1104]

string = '*base'
if '*' in string:
    in_run = string[1:]
    print(in_run)