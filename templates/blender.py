def generate(st):
    prompt1 = st.text_input('Prompt 1')
    prompt2 = st.text_input('Prompt 2')
    return {
        'prompt1': {
            'prompt': prompt1
        },
        'prompt2': {
            'prompt': prompt2
        },
        'image_blend': {
            'prompt': None,
            'image_prompt': '*prompt1||*prompt2'
        }
    }