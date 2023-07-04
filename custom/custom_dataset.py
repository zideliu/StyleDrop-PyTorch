
from torch.utils.data import Dataset

import os
import numpy as np
import taming.models.vqgan
import open_clip
import random
from PIL import Image
import torch
import math
import json
import torchvision.transforms as transforms
torch.manual_seed(0)
np.random.seed(0)

class test_custom_dataset(Dataset):
    
    def __init__(self, style: str = None):
        self.empty_context = np.load("assets/contexts/empty_context.npy")
        self.object=[
            "A chihuahua ",
            "A tabby cat ",
            "A portrait of chihuahua ",
            "An apple on the table ",
            "A banana on the table ",
            "A church on the street ",
            "A church in the mountain ",
            "A church in the field ",
            "A church on the beach ",
            "A chihuahua walking on the street ",
            "A tabby cat walking on the street",
            "A portrait of tabby cat ",
            "An apple on the dish ", 
            "A banana on the dish ", 
            "A human walking on the street ", 
            "A temple on the street ",
            "A temple in the mountain ",
            "A temple in the field ",
            "A temple on the beach ",
            "A chihuahua walking in the forest ",
            "A tabby cat walking in the forest ",
            "A portrait of human face ",
            "An apple on the ground ",
            "A banana on the ground ",
            "A human walking in the forest ",
            "A cabin on the street ",
            "A cabin in the mountain ",
            "A cabin in the field ",
            "A cabin on the beach ",
        ]
        self.style = [
            "in 3d rendering style",
        ]
        if style is not None:
            self.style = [style]
        
    def __getitem__(self, index):
        prompt = self.object[index]+self.style[0]

        return prompt, prompt
    
    def __len__(self):
        return len(self.object)
    
    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v.clamp_(0., 1.)
        return v
    
    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_cc3m_val.npz'
    
    
class train_custom_dataset(Dataset):
    
    def __init__(self, train_file: str=None, ):
        
        self.train_img = json.load(open(train_file, 'r'))
        self.path_preffix = "/".join(train_file.split("/")[:-1])
        self.prompt = []
        self.image = []
        self.style = []
        for im in self.train_img.keys():
            im_path = os.path.join(self.path_preffix, im)
            self.object = self.train_img[im][0]
            self.style = self.train_img[im][1]
            im_prompt = self.object +" "+self.style
            self.image.append(im_path)
            self.prompt.append(im_prompt)
        self.empty_context = np.load("assets/contexts/empty_context.npy")
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        print("-----------------"*3)
        print("train dataset length: ", len(self.prompt))
        print("train dataset length: ", len(self.image))
        print(self.prompt[0])
        print(self.image[0])
        print("-----------------"*3)
    def __getitem__(self, index):
        prompt = self.prompt[0]
        image = Image.open(self.image[0]).convert("RGB")
        image = self.transform(image)
        
        return image,prompt
        # return dict(img=image_embedding, text=text_embedding)
    
    def __len__(self):
        return 24
    
    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v.clamp_(0., 1.)
        return v
    
    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_cc3m_val.npz'
    
    
    
    
    
class  Discriptor(Dataset):
    def __init__(self,style: str=None):
        self.object =[
            # "A parrot ",
            # "A bird ",
            # "A chihuahua in the snow",
            # "A towel ",
            # "A number '1' ",
            # "A number '2' ",
            # "A number '3' ",
            # "A number '6' ",
            # "A letter 'L' ",
            # "A letter 'Z' ",
            # "A letter 'D' ",
            # "A rabbit ",
            # "A train ",
            # "A table ",
            # "A dish ",
            # "A large boat ",
            # "A puppy ",
            # "A cup ",
            # "A watermelon ",
            # "An apple ",
            # "A banana ",
            # "A chair ",
            # "A Welsh Corgi ",
            # "A cat ",
            # "A house ",
            # "A flower ",
            # "A sunflower ",
            # "A car ",
            # "A jeep car ",
            # "A truck ",
            # "A Posche car ",
            # "A vase ",
            # "A chihuahua ",
            # "A tabby cat ",
            "A portrait of chihuahua ",
            "An apple on the table ",
            "A banana on the table ",
            "A human ",
            "A church on the street ",
            "A church in the mountain ",
            "A church in the field ",
            "A church on the beach ",
            "A chihuahua walking on the street ",
            "A tabby cat walking on the street",
            "A portrait of tabby cat ",
            "An apple on the dish ", 
            "A banana on the dish ", 
            "A human walking on the street ", 
            "A temple on the street ",
            "A temple in the mountain ",
            "A temple in the field ",
            "A temple on the beach ",
            "A chihuahua walking in the forest ",
            "A tabby cat walking in the forest ",
            "A portrait of human face ",
            "An apple on the ground ",
            "A banana on the ground ",
            "A human walking in the forest ",
            "A cabin on the street ",
            "A cabin in the mountain ",
            "A cabin in the field ",
            "A cabin on the beach ",
            "A letter 'A' ",
            "A letter 'B' ",
            "A letter 'C' ",
            "A letter 'D' ",
            "A letter 'E' ",
            "A letter 'F' ",
            "A letter 'G' ",
            "A butterfly ",
            " A baby penguin ",
            "A bench ",
            "A boat ",
            "A cow ",
            "A hat ",
            "A piano ",
            "A robot ",
            "A christmas tree ",
            "A dog ",
            "A moose ",
        ]
        
        self.style =[
            "in 3d rendering style",
        ]
        if style is not None:
            self.style = [style]
        
    def __getitem__(self, index):
        prompt = self.object[index]+self.style[0]
        return prompt
    
    def __len__(self):
        return len(self.object)
    
    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v.clamp_(0., 1.)
        return v
    
    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_cc3m_val.npz'
    