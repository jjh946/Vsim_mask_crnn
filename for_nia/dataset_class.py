import numpy as np
from PIL import Image, ImageDraw
import os
import json
import base64
import torch
from torchvision import tv_tensors
from torch.utils.data import Dataset


# 데이터셋 정의
class InstanceSegmentationDataset(Dataset):
    def __init__(self, image_dir, bounding_box_dir, polygon_dir, transform=None):
        self.image_dir = image_dir
        self.bounding_box_dir = bounding_box_dir
        self.polygon_dir = polygon_dir
       
        # 파일 이름 목록을 확장자 없이 저장
        self.filenames = [f[:-4] for f in os.listdir(image_dir) if f.endswith('.png')]
        
        self.transforms = transform

    def __len__(self):
        # 데이터셋의 크기 반환
        return len(self.filenames)

    def __getitem__(self, idx):
        # 이미지와 어노테이션 경로 로드
        image_path = os.path.join(self.image_dir, self.filenames[idx] + '.png')
        bounding_box_path = os.path.join(self.bounding_box_dir, self.filenames[idx] + '.json')
        polygon_path = os.path.join(self.polygon_dir, self.filenames[idx] + '.json')

        # 이미지 열기
        image = Image.open(image_path).convert("RGB")

        # 어노테이션 JSON 파일 열기
        with open(bounding_box_path) as f:
            bounding_boxes = json.load(f)

        with open(polygon_path) as f:
            polygons = json.load(f)

        # 레이블 딕셔너리 정의 (0은 배경이므로 레이블은 1부터 시작)
        label_dict = {"TrafficSigns": 1, "Vehicles": 2}

        # 어노테이션 데이터 가져오기
        box_annotations = bounding_boxes['annotations']
        mask_annotations = polygons['annotations']

        # 폴리곤 포인트 디코딩
        for mask_annotation in mask_annotations:
            points = base64.b64decode(mask_annotation['points'])
            mask_annotation['point_list'] = [
                {'x': points[i], 'y': points[i+1]}
                for i in range(0, len(points), 2)
            ]

        # 폴리곤 포인트를 마스크로 변환
        masks = []
        labels = []
        boxes = []

        for box_annotation, mask_annotation in zip(box_annotations, mask_annotations):  
            # 폴리곤 포인트를 마스크로 변환
            mask_img = Image.new('L', image.size, 0)
            polygon_point_list = [tuple(xy.values()) for xy in mask_annotation['point_list']]
            ImageDraw.Draw(mask_img).polygon(polygon_point_list, outline=1, fill=1)
            mask = np.array(mask_img)
            masks.append(mask)

            # 바운딩 박스를 리스트로 변환
            box = [int(box_annotation['x_min']), int(box_annotation['y_min']), int(box_annotation['x_max']), int(box_annotation['y_max'])]
            boxes.append(box)
            
            # 마스크 레이블
            mask_label = mask_annotation['class']
            labels.append(label_dict[mask_label])

        if masks:
            masks = np.stack(masks, axis=-1)
        else:
            # 어노테이션이 없으면 더미 마스크 생성
            masks = np.zeros((image.shape[0], image.shape[1], 1))

        image = np.array(image).transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.uint8)
        
        masks = masks.transpose(2, 0, 1)
        masks = torch.tensor(masks, dtype=torch.uint8)

        # 다른 어노테이션 정보 가져오기
        image_id = idx
        boxes = torch.as_tensor(boxes, dtype=torch.int32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        
        # 타겟 어노테이션 정보 수집
        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(bounding_boxes['images']['width'], bounding_boxes['images']['height']))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image = self.transforms(image)
            

        return image, target
        #파이토치 텐서 형식으로 변환한 이미지 리턴
        #