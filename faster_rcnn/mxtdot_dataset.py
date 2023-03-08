import torch
import numpy as np
import os
from xml.etree import ElementTree as et
from PIL import Image

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

        iscrowd = torch.zeros((bboxes.shape[0],), dtype=torch.int64)

        # target dictionary
        t = {}
        t["boxes"] = bboxes
        t["labels"] = labels
        t["area"] = area
        t["iscrowd"] = iscrowd
        t["image_id"] = torch.tensor([idx])

        # apply the image transforms
        if self.transforms:
            img, t = self.transforms(image, t)
        return img, t
    
    def __len__(self):
        return len(self.xml_paths)










