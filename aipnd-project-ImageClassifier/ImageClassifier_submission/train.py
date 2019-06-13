import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
import argparse
import json
from Utilities import preprocess_utility
from Utilities import model_utility

parser = argparse.ArgumentParser(
    description='Script to train the image classifier model',
)
parser.add_argument("data_dir", help="The main directory containing all the images")
parser.add_argument("-s", "--save_dir", help="Location for saving model checkpoint")
parser.add_argument("-a", "--arch", help="The pre trained model to be used",default='vgg19')
parser.add_argument("-lr", "--learning_rate", help="The learning rate to be used for training",default=0.0001)
parser.add_argument("-hu", "--hidden_units",  nargs='+', type=int, help="Values representing nodes in each hidden unit", default=[1024,512,256])
parser.add_argument("-e", "--epochs", dest='epochs', action='store', default=5, type=int)
parser.add_argument("-g", "--gpu", action="store_true", help="Use GPU for training")   
args = parser.parse_args() 

def trainSequence():
    print("Inside the main")
### Call function to load and transform the data

    train_dataloader,validate_dataloader,test_dataloader,train_imagefolder = preprocess_utility.loadTransformData(args.data_dir)
    
    classNumber_to_name, class_count = preprocess_utility.loadJSONClassLookup()
    print(f"Completed the class count is {class_count}")
    print(f"GPU val is {args.gpu}")
    model,optimizer,criterion,classifier =         model_utility.createModel(args.arch,args.hidden_units,args.learning_rate,class_count,args.gpu)
    print(f"Completed model create {model}")
    model_utility.trainValidate(model,train_dataloader,validate_dataloader,optimizer,criterion,epochs=args.epochs,gpu=args.gpu)
    print(f"Completed training and validation....") 
    model_utility.saveClassifierModel(model,classifier,args.save_dir,args.arch,optimizer,train_imagefolder,epochs=args.epochs)
    
    

if __name__ == "__main__":
    trainSequence()
    ##Sample train command
    ##python train.py flowers --save_dir chkpointdir --epochs 10 --arch vgg19 --hidden_units 1024 512 256 --gpu 
    ##python train.py flowers --save_dir chkpointdir --epochs 10 --arch vgg13 --hidden_units 1024 512 256 --gpu False
    

    
    
