#importing the required libraries
import torch
import torchvision
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
import torchvision.transforms as transforms
import datetime
import shutil

#replace the below paths with respect to the location of this training script
CURRENT_MODEL_PATH = "model/current_model/alexnet_model_current_prod.pth"
OLD_MODEL_DIR = "model/old_models/"
TRAINING_SAMPLE_DIR = "model/training_samples/"
TRAINED_SAMPLE_DIR = "model/trained_samples/"
IMAGENET_LABELS_FILE = "model/imagenet-classes.txt"

MIN_IMAGES = 4 # setting this low value for testing purpose only

#run this script only where there are more than min_images in training samples.
# we are setting this condition because training model on less number of samples will not be usefull
if len(os.listdir(TRAINING_SAMPLE_DIR)) > MIN_IMAGES:

    #Inheriting Dataset class of Pytorch so that we can create a dataset from our training samples 
    class ImageNet_Dataset(Dataset):
        #the constructor of the class will take 3 parameters
        # img_dir - directory where the training images are placed
        # label_file - directory where the lable file is placed which contains all the labels
        # transform - transformation which will be applied on the images
        def __init__(self, img_dir, label_file, transform = None):
            self.img_dir = img_dir
            self.label_file = label_file
            self.transform = transform
        
        def __len__(self):
            return len(os.listdir(self.img_dir))
        
        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.to_list()
            
            img_name = os.listdir(self.img_dir)[idx]
            
            img_path = os.path.join(self.img_dir, img_name)
            
            img = Image.open(img_path)
            
            img = self.transform(img)
            
            img_label = img_name.split("_label$$")[1].split(".")[0]
            
            #preparing the label list from the file
            label_list = [f.split("\n")[0] for f in open(self.label_file)]
            
            #storing label index
            label_index = label_list.index(img_label)

            #returning the image and its label's index
            return img, label_index
            
    #define transformations to apply on the training samples
    transform = transforms.Compose([            #[1]
    transforms.Resize(256),                    #[2]
    transforms.CenterCrop(224),                #[3]
    transforms.ToTensor(),                     #[4]
    transforms.Normalize(                      #[5]
    mean=[0.485, 0.456, 0.406],                #[6]
    std=[0.229, 0.224, 0.225]                  #[7]
    )])

    #preparing the dataset of the images present in the training samples folder
    img_dataset = ImageNet_Dataset(img_dir = TRAINING_SAMPLE_DIR, label_file = IMAGENET_LABELS_FILE, transform = transform)

    #dataloader from dataset
    dataloaders = DataLoader(img_dataset, batch_size = 16, shuffle = True)

    #function to train the model
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

    #     best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            scheduler.step()

            epoch_loss = running_loss / len(img_dataset)
            epoch_acc = running_corrects.double() / len(img_dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format("train", epoch_loss, epoch_acc))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        return model

    # getting the available device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #declaring the model variable
    model_ft = torchvision.models.alexnet()

    #loading the model from current model
    model_ft.load_state_dict(torch.load(CURRENT_MODEL_PATH))

    #transfer model to the available device 
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    #newly trained model 
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)

    #move the training samples to trained samples folder
    for f in os.listdir(TRAINING_SAMPLE_DIR):
        shutil.move(TRAINING_SAMPLE_DIR+f, TRAINED_SAMPLE_DIR)

    #old_model_name
    now = datetime.datetime.now()
    old_model_name = 'alexnet_model_'+ str(now.strftime("%Y-%m-%d_%H_%M_%S"))+".pth"

    #move the current production model to the old folder
    shutil.move(CURRENT_MODEL_PATH, OLD_MODEL_DIR + old_model_name)

    #save the new model in current_model folder which is our production model
    torch.save(model_ft.state_dict(), CURRENT_MODEL_PATH)