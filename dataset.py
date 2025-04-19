import os
import cv2
from torchvision import transforms
from torch.utils.data import Dataset
import torch

class KvasirSegDataset(Dataset):
    def __init__(self, image_dir, label_dir, device='cuda',transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.device = device
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        #获得图像和label路径
        image_path = os.path.join(self.image_dir, self.image_files[index])
        label_path = os.path.join(self.label_dir, self.label_files[index])
        #cv是bgr，转化为rgb后转化为tensor
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        label = cv2.imread(label_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
        #转换到device
        image = image.to(self.device)
        label = label.to(self.device)

        return image, label

