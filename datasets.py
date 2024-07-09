import glob
import random
import os
import natsort
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