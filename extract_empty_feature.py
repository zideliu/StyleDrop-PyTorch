import os
import numpy as np
import open_clip

def main():
    prompts = [
        '',
    ]

    device = 'cuda'
    model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', 'laion2b_s39b_b160k')
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

    text_tokens = tokenizer(prompts).to(device)
    latent = model.encode_text(text_tokens)

    print(latent.shape)
    c = latent[0].detach().cpu().float().numpy()
    save_dir = f'assets/contexts'
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'empty_context.npy'), c)


if __name__ == '__main__':
    main()
