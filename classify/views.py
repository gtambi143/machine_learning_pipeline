from django.shortcuts import render
from django.http import HttpResponse

#import following packages which will be used to store the file uploaded
from django.conf import settings
from django.core.files.storage import FileSystemStorage

#for copying the file
import shutil

#loading libraries for pytorch
from torchvision import models
import torch
from torchvision import transforms
from PIL import Image #to load the uploaded image

alexnet = models.alexnet(pretrained = True)

# Create your views here.
def index(request):
    dir(models)
    #load the image net classes you need to put the imagenet classes text file in you computer and use its location here
    with open(settings.BASE_DIR+"model\imagenet-classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    return render(request, 'image_upload.html', context = {'imagenet_classes':classes})

def upload_image(request):
    # print("image upload function")
    print(request.FILES['image_file'])
    f = request.FILES['image_file']

    fs = FileSystemStorage()

    filename = fs.save(f.name, f)
    uploaded_file_url = fs.url(filename)

    #predict the class of the input image
    img_class, confidence = predict_image(filename)

    #load the image net classes you need to put the imagenet classes text file in you computer and use its location here
    with open(settings.BASE_DIR+"model\imagenet-classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    return render(request, 'image_upload.html', context = {'uploaded_file_url':uploaded_file_url, 'img_class':img_class, 'confidence': confidence, 'imagenet_classes':classes})

def predict_image(img_path):
    #define the transformations that needs to be apply before passing the image into the model 
    transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

    #below command will load the image and convert it into 3 channel image as the model requires 3 channel image
    img = Image.open(settings.MEDIA_ROOT+'/'+img_path).convert('RGB')
    
    #transformm the image by applying the transformations defined above
    img_t = transform(img)

    #adding one more dimension for batches as model takes the input in batches of images but we have only one image here so we will have one batch containing 1 image
    batch_t = torch.unsqueeze(img_t, 0)

    #put the model in eval mode
    alexnet.eval()

    #getting the output from model
    out = alexnet(batch_t)

    #load the image net classes you need to put the imagenet classes text file in you computer and use its location here
    with open(settings.BASE_DIR+"model\imagenet-classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)
    
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    return (classes[index[0]], percentage[index[0]].item())

def store_training_set(request):
    #image file path
    path = request.POST['file_path']

    #class label of the image
    image_class = request.POST['input_class']

    #now we will save the file into the training sample folder
    name = path.split("/media/")[1]
    filename = name.split(".")[0]
    extension = name.split(".")[1]

    new_filename = filename+"_label$$"+image_class+"."+extension

    #moving file to the training folder
    print(settings.BASE_DIR+"/model/training_samples/"+new_filename)
    shutil.copyfile(settings.BASE_DIR+path, settings.BASE_DIR+"/model/training_samples/"+new_filename)

    #load the image net classes you need to put the imagenet classes text file in you computer and use its location here
    with open(settings.BASE_DIR+"model\imagenet-classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    return render(request, 'image_upload.html', context = {'imagenet_classes':classes})
