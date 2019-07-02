import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image
import utility

def build_model(arch, hidden_units, output_units, gpu):
    '''load pretrained model and substitute classifier'''
    arch = getattr(models, arch)
    model = arch(pretrained = True) #load pretrained model
    for param in model.parameters():
        param.requires_grad = False
    #substitute classifier
    model.classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, output_units),
                                     nn.LogSoftmax(dim=1)
                                     )
    model.to("cuda" if gpu else "cpu")
    return model

def save_model(model, loss, epochs, learnrate, optimizer, class_to_idx, save_dir):
    '''save model, loss function, epochs, learnrate, optimizer and mapping in checkpoint'''
    model.class_to_idx = class_to_idx

    checkpoint = {"batch_size": 32,
                  "model": model,
                  "loss": loss,
                  "epochs": epochs,
                  "learnrate": learnrate,
                  "optimizer_state": optimizer.state_dict}
    torch.save(checkpoint, "checkpoint.pth")

def load_model(path):
    '''load model from checkpoint'''
    checkpoint = torch.load(path)
    return checkpoint["model"]

def train(model, criterion, optimizer, trainloader, validloader, epochs, gpu):
    '''train model'''
    print_every = 50
    for e in range(epochs):
        running_train_loss = 0
        for i,(images, labels) in enumerate(trainloader):
            images, labels = images.to("cuda" if gpu else "cpu"), labels.to("cuda" if gpu else "cpu") #move data to gpu if defined
            logps = model.forward(images) #forward propagation
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward() #backward propagation
            optimizer.step()

            running_train_loss += loss.item()

            #evaluate model
            if (i+1)%print_every == 0:
                model.eval()
                running_accuracy, running_valid_loss = 0, 0
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to("cuda" if gpu else "cpu"), labels.to("cuda" if gpu else "cpu") #move data to gpu if defined
                        logps = model.forward(images)
                        ps = torch.exp(logps)
                        probs, predictions = ps.topk(1, dim = 1)

                        running_valid_loss += criterion(logps, labels).item()

                        equals = predictions[:,-1] == labels
                        running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                accuracy = running_accuracy/len(validloader)
                valid_loss = running_valid_loss/len(validloader)
                train_loss = running_train_loss/print_every
                model.train()
                running_train_loss = 0
                print("Epoch: {}, Batch: {}/{}, Trainloss: {}, Validationloss: {}, Accuracy: {}"
                            .format(e, (i+1), len(trainloader), train_loss, valid_loss, accuracy))

def predict(image_path, model, topk, gpu):
    '''predict top k classes for a given model and image'''
    model.to("cuda" if gpu else "cpu") #move model to gpu if defined
    model.eval()
    with torch.no_grad():
        img = utility.process_image(image_path).unsqueeze(0).type(torch.FloatTensor).to("cuda" if gpu else "cpu") #move image to gpu if defined
        logps = model.forward(img)
        ps = torch.exp(logps)
        ps, idxs = ps.topk(topk, dim = 1)
        idx_to_class = dict(map(reversed, model.class_to_idx.items()))
        preds = [idx_to_class[idx.item()] for idx in idxs[-1,:]]
    return ps, preds
