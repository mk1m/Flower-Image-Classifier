import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image

#Jupyter Notebook -> train.py
def load_data(path):
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    training_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    testing_dataset = datasets.ImageFolder(test_dir ,transform = test_transforms) 
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    trainloader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_dataset, batch_size=64)
    testloader = torch.utils.data.DataLoader(testing_dataset, batch_size=64)

    return training_dataset, trainloader, validationloader, testloader
    
#Build Model
def build(architecture, hidden_units):
    model = models.vgg16(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(
              nn.Linear(25088, hidden_units),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(hidden_units, 256),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(256, 102),
              nn.LogSoftmax(dim = 1)
            )
    model.classifier = classifier
    return model

#train model
def train(model, epochs, learning_rate, trainloader, validationloader):
    steps = 0
    print_every = 5
    running_loss = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.003)

    model.to(device)    
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                validation_loss = 0
                validation_accuracy = 0
                model.eval()
            
                with torch.no_grad():
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                    
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        validation_loss += batch_loss.item()

                    # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}, "
                    f"Train loss: {running_loss/print_every:.3f}, "
                    f"Validation loss: {validation_loss/len(validationloader):.3f}, "
                    f"Validation accuracy: {validation_accuracy/len(validationloader):.3f}")
            
                running_loss = 0
                model.train()
    return model, criterion
    
#test model
def test(model, testloader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_loss = 0
    test_accuracy = 0
    model.eval() 
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

        # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(testloader):.3f}, "
        f"Test accuracy: {test_accuracy/len(testloader):.3f}")
    print("Done")
    
def save(model, architecture, hidden_units, epochs, learning_rate, save_dir):
    checkpoint = {
        'epochs': epochs,
        'learning_rate': 0.003,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'criterion': criterion.state_dict(),
        'class_to_idx': training_dataset.class_to_idx
    }

    torch.save(checkpoint, 'checkpoint.pth')

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

#Command Line
parser = argparse.ArgumentParser(description='Training a neural network')
parser.add_argument('data_directory', help='Dataset')
parser.add_argument('--save_dir', help='Checkpoint path')
parser.add_argument('--arch', help='Network architecture: "vgg16"')
parser.add_argument('--learning_rate', help='Learning rate')
parser.add_argument('--hidden_units', help='Hidden units')
parser.add_argument('--epochs', help='Number of epochs')
parser.add_argument('--gpu', help='Use GPU', action='store_true')


args = parser.parse_args()


save_dir = '' if args.save_dir is None else args.save_dir
arch = 'vgg16' if args.arch is None else args.arch
learning_rate = 0.003 if args.learning_rate is None else int(args.learning_rate)
hidden_units = 512 if args.hidden_units is None else float(args.hidden_units)
epochs = 5 if args.epochs is None else int(args.epochs)
gpu = args.gpu

training_dataset, trainloader, validationloader, testloader = load_data(args.data_directory)

model = build(arch, hidden_units)
model.class_to_idx = training_dataset.class_to_idx

model, criterion = train(model, epochs, learning_rate, trainloader, validationloader)
test(model, testloader, criterion)
save(model, arch, hidden_units, epochs, learning_rate, save_dir)