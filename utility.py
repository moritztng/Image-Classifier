import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
from PIL import Image

def load_transform_data(path):
    '''load and transform the data'''
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #compose transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])
                                          ])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])
                                          ])
    
    #load and transform data
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)

    trainloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    validloader = DataLoader(valid_dataset, batch_size = 32, shuffle = True)
    testloader = DataLoader(test_dataset, batch_size = 32, shuffle = True)
    
    return trainloader, validloader, testloader, train_dataset.class_to_idx

def load_mapping(path):
    '''load class to name mapping from json file'''
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def process_image(img_path):
    '''resize, crop and normalize image. Return tensor. '''
    with Image.open(img_path) as img:
        #resize -> min length = 256px
        scale = min(img.size)/256
        width, height = img.size
        img = img.resize((int(width/scale), int(height/scale)))
        
        #crop image to 224 x 224
        width, height = img.size 
        new_width, new_height = 224, 224
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        img = img.crop((left, top, right, bottom))
        
        #normalize image
        np_img = np.array(img)
        np_img = np_img / 255
        np_img = (np_img-[0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
        np_img = np_img.transpose(2,0,1)
    
    return torch.from_numpy(np_img)
