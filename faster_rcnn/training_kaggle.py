
import os
import random
import numpy as np
import pandas as pd
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2
from PIL import Image

# xml library for parsing xml files
from xml.etree import ElementTree as et

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans  
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# these are the helper libraries imported.
from engine import train_one_epoch, evaluate
import utils
import transforms as T

# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import config_server as config

# for use with cv2 (ToTensorV2 converts image to pytorch tensor without div by 255)
def get_transform_album(train):
    
    if train:
        return A.Compose([
                            A.HorizontalFlip(0.5),
                     # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2(p=1.0) 
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# for use with PIL (PILToTensor scales img pixel values to range [0,1])
def get_transform_pil(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class MxtDotDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path_imgs, folder_path_xmls, classes, transforms=None):
        self.transforms = transforms

        self.folder_path_imgs = folder_path_imgs
        self.folder_path_xmls = folder_path_xmls
        self.classes = classes

        self.xml_paths = [os.path.join(self.folder_path_xmls,f) for f in os.listdir(self.folder_path_xmls) if f.split(".")[-1]=="xml"]
        self.xml_names = sorted([p.split("/")[-1] for p in self.xml_paths])
        # self.xml_paths, self.xml_names = zip(*sorted(list(zip(self.xml_paths,self.xml_names)), key = lambda x: x[1]))

    def __getitem__(self, idx):
        xml_name = self.xml_names[idx]
        xml_path = os.path.join(self.folder_path_xmls,xml_name)
        

        image_name = xml_name[:-3] + "jpg"
        image_path = os.path.join(self.folder_path_imgs,image_name)
        
        image = Image.open(image_path).convert("RGB")

        bboxes = []
        labels = []

        tree = et.parse(xml_path)
        root = tree.getroot()

        # image_width, image_height = image.size

        for member in root.findall("object"):
            if member.find("name").text != "QUAD":
                labels.append(self.classes.index(member.find("name").text))

                xmin = int(member.find("bndbox").find("xmin").text)
                ymin = int(member.find("bndbox").find("ymin").text)
                xmax = int(member.find("bndbox").find("xmax").text)
                ymax = int(member.find("bndbox").find("ymax").text)
                
                """
                CHANGED FOR POTHOLES DATASET
                """
                
                if xmin >= xmax:
                    xmax = xmin + 1
                
                if ymin >= ymax:
                    ymax = ymin + 1

                bboxes.append([xmin,ymin,xmax,ymax])
        
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        area = (bboxes[:,3]-bboxes[:,1]) * (bboxes[:,2]-bboxes[:,0])
        # print(f"bboxes[:,3]: {bboxes[:,3]}")
        # print(f"bboxes[:,1]: {bboxes[:,1]}")
        # print(f"bboxes[:,3]-bboxes[:,1]: {bboxes[:,3]-bboxes[:,1]}")
        # print(f"bboxes[:,3]: {bboxes[:,2]}")
        # print(f"bboxes[:,1]: {bboxes[:,0]}")
        # print(f"bboxes[:,2]-bboxes[:,0]: {bboxes[:,2]-bboxes[:,0]}")
        # print(f"area: {area}")
        iscrowd = torch.zeros((bboxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # print(f"iscrowd: {iscrowd}")
        # print(f"bboxes.shape[0]: {bboxes.shape[0]}")
        # target dictionary
        t = {}
        t["boxes"] = bboxes
        t["labels"] = labels
        t["area"] = area
        t["iscrowd"] = iscrowd
        t["image_id"] = torch.tensor([idx])

        # apply the image transforms
        if self.transforms:
            # img,t = self.transforms(image,t)
            img, t = self.transforms(image, t)
        return img, t
    
    def __len__(self):
        return len(self.xml_paths)

class FruitImagesDatasetPIL(torch.utils.data.Dataset):

    def __init__(self, files_dir, width, height, classes, transforms=None):
        self.transforms = transforms
        self.files_dir = files_dir
        self.height = height
        self.width = width
        
        # sorting the images for consistency
        # To get images, the extension of the filename is checked to be jpg
        self.imgs = [image for image in sorted(os.listdir(files_dir))
                        if image[-4:]=='.jpg']
        
        
        # classes: 0 index is reserved for background
        # self.classes = [_, 'apple','banana','orange']
        self.classes = classes

    def __getitem__(self, idx):

        img_name = self.imgs[idx]
        image_path = os.path.join(self.files_dir, img_name)

        # reading the images and converting them to correct size and color    
        # img = cv2.imread(image_path)
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        # img_res /= 255.0

        img = Image.open(image_path).convert("RGB")
        img_res = img.resize((self.width, self.height), Image.LANCZOS)
        
        # annotation file
        annot_filename = img_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.files_dir, annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # cv2 image gives size as height x width
        # wt = img.shape[1]
        # ht = img.shape[0]
        wt, ht = img.size
        
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            
            # bounding box
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            
            
            xmin_corr = (xmin/wt)*self.width
            xmax_corr = (xmax/wt)*self.width
            ymin_corr = (ymin/ht)*self.height
            ymax_corr = (ymax/ht)*self.height
            
            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])
        
        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        labels = torch.as_tensor(labels, dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id


        if self.transforms:
            img_res, target = self.transforms(img_res, target)
            # sample = self.transforms(image = img_res,
            #                          bboxes = target['boxes'],
            #                          labels = labels)
            
            # img_res = sample['image']
            # target['boxes'] = torch.Tensor(sample['bboxes'])
            
            
            
        return img_res, target

    def __len__(self):
        return len(self.imgs)



class FruitImagesDatasetCV2(torch.utils.data.Dataset):

    def __init__(self, files_dir, width, height, classes, transforms=None):
        self.transforms = transforms
        self.files_dir = files_dir
        self.height = height
        self.width = width
        
        # sorting the images for consistency
        # To get images, the extension of the filename is checked to be jpg
        self.imgs = [image for image in sorted(os.listdir(files_dir))
                        if image[-4:]=='.jpg']
        
        
        # classes: 0 index is reserved for background
        # self.classes = [_, 'apple','banana','orange']
        self.classes = classes

    def __getitem__(self, idx):

        img_name = self.imgs[idx]
        image_path = os.path.join(self.files_dir, img_name)

        # reading the images and converting them to correct size and color    
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0
        
        # annotation file
        annot_filename = img_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.files_dir, annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # cv2 image gives size as height x width
        wt = img.shape[1]
        ht = img.shape[0]
        
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            
            # bounding box
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            
            
            xmin_corr = (xmin/wt)*self.width
            xmax_corr = (xmax/wt)*self.width
            ymin_corr = (ymin/ht)*self.height
            ymax_corr = (ymax/ht)*self.height
            
            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])
        
        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        labels = torch.as_tensor(labels, dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id


        if self.transforms:
            
            sample = self.transforms(image = img_res,
                                     bboxes = target['boxes'],
                                     labels = labels)
            
            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
            
            
        return img_res, target

    def __len__(self):
        return len(self.imgs)

def get_model_fasterrcnn(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # # now get the number of input features for the mask classifier
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # # and replace the mask predictor with a new one
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
    #                                                    hidden_layer,
    #                                                    num_classes)

    return model

# # check dataset
# dataset = FruitImagesDatasetCV2(config.FILES_DIR, 224, 224, classes = config.CLASSES)
# print('length of dataset = ', len(dataset), '\n')

# # getting the image and target for a test index.  Feel free to change the index.
# img, target = dataset[78]
# print(img.shape, '\n',target)


# use our dataset and defined transformations
# dataset = FruitImagesDatasetCV2(config.FILES_DIR, 480, 480, classes = config.CLASSES, transforms= get_transform_album(train=True))
# dataset_test = FruitImagesDatasetCV2(config.FILES_DIR, 480, 480, classes = config.CLASSES, transforms= get_transform_album(train=False))

# dataset = FruitImagesDatasetPIL(config.FILES_DIR, 480, 480, classes = config.CLASSES, transforms= get_transform_pil(train=True))
# dataset_test = FruitImagesDatasetPIL(config.FILES_DIR, 480, 480, classes = config.CLASSES, transforms= get_transform_pil(train=False))

# dataset = MxtDotDataset(config.FILES_DIR, config.FILES_DIR, config.CLASSES, transforms= get_transform_pil(train=True))
# dataset_test = MxtDotDataset(config.FILES_DIR, config.FILES_DIR, config.CLASSES, transforms= get_transform_pil(train=False))

dataset = MxtDotDataset(config.TRAIN_DIR_IMGS, config.TRAIN_DIR_XMLS, config.CLASSES, transforms= get_transform_pil(train=True))
dataset_test = MxtDotDataset(config.TEST_DIR_IMGS, config.TEST_DIR_XMLS, config.CLASSES, transforms= get_transform_pil(train=False))
# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

# train test split
test_split = 0.2
tsize = int(len(dataset)*test_split)
# dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
# dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)



# to train on gpu if selected.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


num_classes = config.NUM_CLASSES

# get the model using our helper function
model = get_model_fasterrcnn(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=config.LEARNING_RATE,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=config.SCHEDULER_STEP_SIZE,
                                               gamma=config.SCHEDULER_GAMMA)

# training for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    # training for one epoch
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)