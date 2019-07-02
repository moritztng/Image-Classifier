from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import torch
import utility, learning

#init parser
parser = ArgumentParser(description="Make predictions with a trained neural network")
parser.add_argument("img_path", type=str, help="path to the image")
parser.add_argument("checkpoint_path", type=str, help="path to the checkpoint")
parser.add_argument("--mapping_path", type=str, default="", help="path to the checkpoint (Default: No_Mapping)")
parser.add_argument("--top_k", type=int, default="3", help="path to the checkpoint (Default: 3)")
parser.add_argument('--gpu', action='store_true', help="Use GPU for training (Default: False)")
args = parser.parse_args()

#load name mapping from json file
cat_to_name = utility.load_mapping(args.mapping_path) if args.mapping_path!="" else {}

#load model from checkpoint
model = learning.load_model(args.checkpoint_path)

#predict classes and propabilities
ps, preds = learning.predict(args.img_path, model, args.top_k, args.gpu)

#map classes to names
preds = [cat_to_name[pred] for pred in preds] if cat_to_name else preds

#print results
for name, prob in zip(preds, ps[-1,:]):
    print("Class: {}, Propability: {}%".format(name, prob*100))
