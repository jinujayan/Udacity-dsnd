import torch
import torch.nn.functional as F
import argparse
import json
from torch import nn
from torch import optim
from torchvision import datasets, transforms,models
from operator import itemgetter
from Utilities import model_utility
from Utilities import preprocess_utility
parser = argparse.ArgumentParser(
    description='Script to predict the class probabilities given a image',
)
parser.add_argument("image_path", help="path to the image being tested")
parser.add_argument("checkpoint", help="path to the saved model checkpoint")
parser.add_argument("-t", "--top_k", help="to list the top k predicted class",type=int, default=5)
parser.add_argument("-c", "--category_names",help="JSON File containing the flower name to class lookup")
parser.add_argument("-g", "--gpu", action="store_true", help="Use GPU for inference")
args = parser.parse_args() 

def predictSequence():
    model = model_utility.loadClassifierModel(args.checkpoint)
    print("Model loaded successfully")
    #processed_image = preprocess_utility.predictImagePreprocessor(args.image_path, 224)
    #print(model)
    if args.category_names:
        print("Predict on flower names...")
        pred_class = model_utility.predictImageClass(args.image_path, model,args.gpu, topk=args.top_k,category_names=args.category_names)
        category = "Flower"
    else:
        category = "Class Number"
        print("Predict on class numbers...")
        pred_class = model_utility.predictImageClass(args.image_path, model,args.gpu, topk=args.top_k)
    
    print("End of prediction")
    print(pred_class)
    print("sorted list....")
    pred_class.sort(key=itemgetter(0), reverse=True)
    print(f"-------------Identified {category}---------------")
    print(f"            {pred_class[0][1]}               ")
    print("---------------------------------------------")
if __name__ == "__main__":
    predictSequence()
    ###Sample run command 
    ###python predict.py flowers/test/5/image_05166.jpg  chkpointdir_test/ImageClassifier_vgg13.pth.tar --category_names cat_to_name.json --gpu
    ##python predict.py flowers/test/5/image_05166.jpg  chkpointdir/ImageClassifier_vgg19.pth.tar --category_names cat_to_name.json --gpu