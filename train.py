from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import utility, learning

#init parser
parser = ArgumentParser(description="Train neural networks on an image dataset!")
parser.add_argument('data_dir', type=str, help="Directory of the images")
parser.add_argument('--save_dir', type=str, default="", help="Directory to save the model (Default: Current Directory)")
parser.add_argument('--arch', type=str, default="vgg19", help="Neural Network Architecture (Default: vgg19)")
parser.add_argument('--learning_rate', type=float, default="0.001", help="Learningrate (Default: 0.001)")
parser.add_argument('--hidden_units', type=int, default="4096", help="Number of hidden units (Default: 4096)")
parser.add_argument('--epochs', type=int, default="3", help="Number of epochs (Default: 3)")
parser.add_argument('--gpu', action='store_true', help="Use GPU for training (Default: False)")
args = parser.parse_args()

#load and transform data
trainloader, validloader, testloader, class_to_idx = utility.load_transform_data(args.data_dir)

#build model, loss function, and optimizer
model = learning.build_model(args.arch, args.hidden_units, len(class_to_idx), args.gpu)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)

#train model
learning.train(model, criterion, optimizer, trainloader, validloader, args.epochs, args.gpu)

#save checkpoint
learning.save_model(model, args.arch, criterion, args.epochs, args.learning_rate, optimizer, class_to_idx, args.save_dir)
