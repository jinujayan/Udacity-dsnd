import torch
from torch import nn
from torch import optim
from torchvision import transforms,models
import torch.nn.functional as F
import datetime
import os
import json
from Utilities import preprocess_utility


class Network(nn.Module):
    def __init__(self, input_length, output_length, hidden_layers, drop_prob=0):
        ''' To Create a Neural network with configurable number of hidden layers,input and      output lengths.
        
            input_length: size of the input(int) 
            output_length: size of the output(int) 
            hidden_layers: A list Containing the nodes in each hidden layer, starting with input layer
            drop_prob: Dropout probability, value between 0-1 (float)
        '''
        super().__init__()
        # Add the first layer
        print(f"Inside init DBG1-> {hidden_layers}")
        self.hidden_layers = nn.ModuleList([nn.Linear(input_length, 
                                                      hidden_layers[0])])
        print("Inside init DBG2")
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        print(f"Inside init DBG2 {layer_sizes}")
        self.hidden_layers.extend([
            nn.Linear(h_input, h_output) for h_input, h_output in layer_sizes])
        
        # Add output layer
        self.output = nn.Linear(hidden_layers[-1], output_length)
        
        # Include dropout
        self.dropout = nn.Dropout(p=drop_prob)
        
    def forward(self, x):
        # Forward through each hidden layer with ReLU and dropout
        for layer in self.hidden_layers:
            x = F.relu(layer(x)) # Apply activation
            x = self.dropout(x) # Apply dropout
        
        # Pass through output layer
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)
    
def createModel(model_arch, hidden_layers,learning_rate,output_length,gpu=True):
    print(f"In create model...show the hidden layers {hidden_layers}")
    if model_arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif model_arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        #print(f" {model_arch} is unsupported model by this application, try vgg16,vgg19 or vgg13")
        raise ValueError(f" {model_arch} is unsupported model by this application, try vgg16,vgg19 or vgg13")
    for param in model.parameters():
        param.requires_grad = False
    
    ###Defining the classifier and tunable params
    vgg_clasifier_input = model.classifier[0].in_features
    
    classifier = Network(input_length=vgg_clasifier_input, 
                     output_length=output_length, 
                     hidden_layers=hidden_layers,drop_prob=0.2)
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model,optimizer, criterion,classifier

def trainValidate(model,train_dataloader,validate_dataloader,optimizer,criterion,epochs=1,gpu=True):
    print(f"Incoming gpu value is {gpu}")
    print(f"Start of training ----------- {datetime.datetime.now()} ----------- ")
    
    steps = 0
    running_loss = 0
    print_every = 25
    if gpu == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using GPU?", torch.cuda.is_available())
    else:
        device = "cpu"
        print(f"GPU not requested...continue with CPU")
    model.to(device)
    for epoch in range(epochs):
        for inputs, labels in train_dataloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validate_dataloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        val_out = model.forward(inputs)
                        batch_loss = criterion(val_out, labels)
                    
                        val_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        val_out_actual = torch.exp(val_out)
                        top_p, top_class = val_out_actual.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {val_loss/len(validate_dataloader):.3f}.. "
                  f"Validation accuracy percent: {accuracy*100/len(validate_dataloader):.3f}")
            
            ##reset running loss, to capture the loss for the duration of next <print_every> steps
                running_loss = 0
                model.train()
    print(f"End of training ------------ {datetime.datetime.now()} ----------- ")        
    return ""
def executeTest(model, test_data, criterion, gpu):
    test_loss = 0
    accuracy = 0
       
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using GPU?", torch.cuda.is_available())
    model.to(device)
    #print(f"The total number of test batch is -> {len(test_data)}")
    for images, labels in test_data:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        #print(f"The test loss in run {i} is {test_loss}")
        # Convert back to softmax distribution
        ps = torch.exp(output)
        # Compare highest prob predicted class ps.max(dim=1)[1] with labels
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        
        accuracy += torch.mean(equals.type(torch.FloatTensor))
            
    return test_loss, accuracy*100/(len(test_data))

def saveClassifierModel(model,classifier,save_path,arch,optimizer,train_imagefolder,epochs):
    if save_path is None:
        save_path = os.getcwd()
        print(f"No save path specified, saving in current directory {save_path}")
    # Create directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
        # Create dictionary of parameters
    param_dict = {
        'arch': arch,
        'model_state_dict': model.state_dict(),
        'class_to_idx': train_imagefolder.class_to_idx,
        'optimizer': optimizer,
        'epochs': epochs,
        'classifier':classifier
        }
    checkpoint_file = "ImageClassifier_"+arch+".pth.tar"
    
    #print(param_dict)
    print(f"The file path being saved is {os.path.join(save_path, checkpoint_file)}")
    print("$$$$$$$$$$$$-----Use this checkpoint file for prediction-----$$$$$$$$$$$$")
    torch.save(param_dict, os.path.join(save_path, checkpoint_file))
    return  ""
          
          
def loadClassifierModel(checkpointpath, gpu=False):
    if gpu:
        print("Using GPU")
        maplocation = "cuda"
    else:
        maplocation = "cpu"
    checkpoint = torch.load(checkpointpath,map_location=maplocation)
    ##model = TheModelClass(*args, **kwargs)
              
    if checkpoint['arch'] == 'vgg13':     
        model = models.vgg13(pretrained=True)
    elif checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        raise ValueError(f" {checkpoint['arch']} is unsupported model by this application, try vgg16,vgg19 or vgg13") 
    model.classifier = checkpoint['classifier']
    model.load_state_dict = checkpoint['model_state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def predictImageClass(image_path, model, gpu,topk=5, category_names=None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #test_im = Image.open(image_path)
    image_array = preprocess_utility.predictImagePreprocessor(image_path,256)
    tensor_image = torch.from_numpy(image_array).float()
    ##Adding a new batch dimension to the single image
    tensor_image.unsqueeze_(0)
        
    if gpu:
        # Check GPU availability
        print("Using GPU")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
        print(f"GPU not requested....continue with CPU")
    tensor_image = tensor_image.to(device)
        
    model.eval()
    model.to(device)
    with torch.no_grad():
        output = model.forward(tensor_image)
        ##Get Probabilities
        ps = torch.exp(output)
        probs, indxes = torch.topk(ps, topk)
        ##On CPU for numpy 
        probs = probs.cpu()
        indxes = indxes.cpu()
        ##Converting to numpy aray for easier manipulations
        indxes = indxes.numpy()
        probs=probs.numpy()
               
        probindex = zip(probs[0], indxes[0])
    class_probs=[]
    flower_probs=[]         
    print(f"The category names is....{category_names}")              
    if category_names:
        with open('cat_to_name.json', 'r') as f:
            classNumber_to_name = json.load(f)
         
        for prob,indx in zip(probs[0], indxes[0]):
            pred_class = [classN  for (classN, m_indx) in model.class_to_idx.items() if m_indx == indx]
            #class_probs.append((prob,pred_class[0])) 
            #print(f"show the class number identified....{pred_class}")
            flower = classNumber_to_name[pred_class[0]]
            print(f"The pred class, flower and probability is ....{pred_class} {flower} {prob}")
            #print(f"show the Flower name identified....{classNumber_to_name[pred_class[0]]}")
            flower_probs.append((prob,flower))
        return flower_probs
    else:
        for prob,indx in zip(probs[0], indxes[0]):
            pred_class = [classN  for (classN, m_indx) in model.class_to_idx.items() if m_indx == indx]
            class_probs.append((prob,pred_class[0])) 
            print(f"The pred class, flower and propability is ....{pred_class} {pred_class[0]} {prob}")
            #print(f"show the class number identified....{pred_class}")
            #flower = classNumber_to_name[pred_class[0]]
            #print(f"show the Flower name identified....{classNumber_to_name[pred_class[0]]}")
            #flower_probs.append((prob,flower))
        return class_probs
        
          
    
    
    

            
            