import os
import numpy as np
from pathlib import Path

import open_clip

def main():
    prompts = [
        'A green train is coming down the tracks.',
        'A group of skiers are preparing to ski down a mountain.',
        'A small kitchen with a low ceiling.',
        'A group of elephants walking in muddy water.',
        'A living area with a television and a table.',
        'A road with traffic lights, street lights and cars.',
        'A bus driving in a city area with traffic signs.',
        'A bus pulls over to the curb close to an intersection.',
        'A group of people are walking and one is holding an umbrella.',
        'A baseball player taking a swing at an incoming ball.',
        'A city street line with brick buildings and trees.',
        'A close up of a plate of broccoli and sauce.',
    ]

    device = 'cuda'
    model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', 'laion2b_s39b_b160k')
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

    text_tokens = tokenizer(prompts).to(device)
    latent = model.encode_text(text_tokens)

    save_dir = Path(f'assets/contexts/run_vis')
    save_dir.mkdir(exist_ok=True, parents=True)
    for i in range(len(latent)):
        c = latent[i].detach().cpu().float().numpy()
        np.save(os.path.join(save_dir, f'{i}.npy'), c)


if __name__ == '__main__':
    main()
