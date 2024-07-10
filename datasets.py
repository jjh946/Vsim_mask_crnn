import base64
import glob
import json
import os
import cv2
import natsort
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_ = None):
        self.transform = transforms.Compose(transforms_)
        self.files_A = sorted(glob.glob(os.path.join(root ) + '*.*'))        
        self.files_A = natsort.natsorted(self.files_A)   
        # json 파일 경로를 받아 파싱한 후 , 주석 데이터를 인스턴스 변수로 저장하는 코드 필요 

    

    def __getitem__(self, index):
        img = Image.open(self.files_A[index])

        if img.mode == 'RGBA':
            img = img.convert('RGB')
        item_A = self.transform(img)
        return {'A': item_A}
        
    def __len__(self):
        return len(self.files_A)
    


class CustomDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, '*.png')))
        self.files = natsort.natsorted(self.files)
        self.annotations = self.load_annotations(root)
        
    def load_annotations(self, root):
        annotations = {}
        annotation_types = ['boundingbox2d', 'polygon']
        for ann_type in annotation_types:
            for ann_file in glob.glob(os.path.join(root, ann_type, '*.json')):
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                    if data['images']['identifier'] not in annotations:
                        annotations[data['images']['identifier']] = []
                    for ann in data['annotations']:
                        annotations[data['images']['identifier']].append((ann_type, ann))
        return annotations

    def __getitem__(self, index):
        img_path = self.files[index]
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        item_A = self.transform(img)
        
        img_id = os.path.basename(img_path)
        annotation = self.annotations.get(img_id, [])
        
        boxes = []
        labels = []
        masks = []
        for ann_type, ann in annotation:
            if ann_type == 'boundingbox2d':
                boxes.append([int(ann['x_min']), int(ann['y_min']), int(ann['x_max']), int(ann['y_max'])])
                labels.append(int(ann['class']))  # Ensure class is int
            elif ann_type == 'polygon':
                mask = self.create_mask_from_polygon(ann, img.size)
                masks.append(mask)
            # Add more conditions for other annotation types if needed

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'masks': torch.stack(masks) if masks else torch.empty(0)
        }
        
        return {'A': item_A, 'target': target}
        
    def __len__(self):
        return len(self.files)
    
    def create_mask_from_polygon(self, ann, img_size):
        width, height = img_size
        mask = np.zeros((height, width), dtype=np.uint8)
        points = base64.b64decode(ann['points'])
        points = np.frombuffer(points, dtype=np.uint16).reshape(-1, 2)
        points = points.astype(np.int32)
        cv2.fillPoly(mask, [points], 1)
        return torch.tensor(mask, dtype=torch.uint8)