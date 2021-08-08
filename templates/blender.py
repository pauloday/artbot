def generate(st):
    prompt1 = st.text_input('Prompt 1')
    prompt2 = st.text_input('Prompt 2')
    return {
        'prompt1': {
            'iterations': 200,
            'images': 10,
            'prompt': prompt1
        },
        'prompt2': {
            'prompt': prompt2
        },
        'image_blend': {
            'prompt': None,
            'image_prompt': '*prompt1||*prompt2'
        },
        'one_two': {
            'image_prompt': None,
            'prompt': f'{prompt1} {prompt2}'
        },
        'two_one': {
            'prompt': f'{prompt2} {prompt1}'
        },
        'one_and_two': {
            'prompt': f'{prompt1}||{prompt2}'
        },
        'one_to_two11': {
            'prompt': f'{prompt1}--{prompt2}'
        },
        'one_to_two12': {
            'prompt': f'{prompt1}--{prompt2}__2'
        },
        'one_to_two21': {
            'prompt': f'{prompt1}__2--{prompt2}'
        },
        'two_to_one11': {
            'prompt': f'{prompt1}--{prompt2}'
        },
        'two_to_one12': {
            'prompt': f'{prompt1}--{prompt2}__2'
        },
        'two_to_one21': {
            'prompt': f'{prompt1}__2--{prompt2}'
        },
        'one_text_two_image': {
            'prompt': prompt1,
            'image_propmt': '*prompt2'
        },
        'one_image_two_text': {
            'prompt': prompt2,
            'image_propmt': '*prompt1'
        },
        'one_start': {
            'iterations': 60,
            'images': 3,
            'image_prompt': None,
            'init_image': '*prompt2'
        },
        'two_start': {
            'prompt': prompt1,
            'init_image': '*prompt2'
        }
    }