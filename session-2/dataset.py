import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import kagglehub
from torchvision import transforms, datasets

class MyDataset(Dataset):

    def __init__(self, images_path, labels_path, transform=None):
        super().__init__()
        
        self.transform = transform
        self.images_path = images_path
        #self.labels_path = labels_path
            
        print("Path to images:", images_path)
        print("Path to labels:", labels_path)

        self.labels_df = pd.read_csv(labels_path)

       

    def __len__(self):
        return len(self.labels_df)


    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        #print("row:", row)
        img_path = os.path.join(self.images_path, "input_" + str(row['suite_id']) + "_" + str(row['sample_id']) + "_" + str(row['code']) + ".jpg")
        #print("img_path:", img_path)
        label = int(row['code']) - 1
        
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        # Return label as torch.long for classification losses like CrossEntropyLoss
        return image, torch.tensor(label, dtype=torch.long)

# Download latest version
#path = kagglehub.dataset_download("gpreda/chinese-mnist")

#print("Path to dataset files:", path)

#data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
#my_dataset = MyDataset(path + "/data/data",path + "/chinese_mnist.csv",transform=data_transforms)

#print("salidaaaa")
#print(my_dataset.__getitem__(1))

