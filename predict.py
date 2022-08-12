import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image


def load_check(filepth):
    model = models.vgg16(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(25088, 512),
                              nn.ReLU(),
                              nn.Dropout(p=0.2),
                              nn.Linear(512, 256),
                              nn.ReLU(),
                              nn.Dropout(p=0.2),
                              nn.Linear(256, 102),
                              nn.LogSoftmax(dim=1))
    model.classifier = classifier
    
    checkpoint = torch.load(filepth)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx'] 
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    return test_transforms(Image.open(image))

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    classes = []
    probs = []
    #turn off gradients
    model.eval()
    with torch.no_grad():
        logps = model.forward(process_image(image_path).unsqueeze(0))
        ps = torch.exp(logps)
        probs, labels = ps.topk(topk, dim=1)
        
        conversion_dict = {model.class_to_idx[i]: i for i in model.class_to_idx}
        classes = [conversion_dict[label] for label in labels.numpy()[0]]
        probs = probs.numpy()[0]
    
        return probs, classes
    
    
#Command Line
parser = argparse.ArgumentParser(description='Predicting flower name and probability')
parser.add_argument('image_path', help='Image path')
parser.add_argument('checkpoint', help='checkpoint')
parser.add_argument('--top_k',  help='Top k most likely')
parser.add_argument('--category_names', help='Category names')
parser.add_argument('--gpu', help='Use GPU', action='store')


args = parser.parse_args()

top_k = 3 if args.top_k is None else int(args.top_k)
category_names = "cat_to_name.json" if args.category_names is None else args.category_names
gpu = args.gpu

model = load_check(args.checkpoint)
print(model)

probs, predict_classes = predict(process_image(args.image_path), model, top_k)

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

classes = []
    
for pred_class in predict_classes:
    classes.append(cat_to_name[pred_class])
