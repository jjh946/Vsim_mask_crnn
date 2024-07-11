import os
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# dataset_class.py에서 InstanceSegmentationDataset 클래스를 가져옵니다.
from dataset_class import InstanceSegmentationDataset

# 테스트를 위한 이미지와 어노테이션 디렉토리 경로 설정
image_dir = '../images'
bounding_box_dir = '../../Saved/1bea28d9bb1d3d9acd7b5e420719f05a/labelingData/Car/N01S01M01/Design0008/outputJson/boundingbox2d'
polygon_dir = '../../Saved/1bea28d9bb1d3d9acd7b5e420719f05a/labelingData/Car/N01S01M01/Design0008/outputJson/polygon'

# 데이터셋 인스턴스 생성
dataset = InstanceSegmentationDataset(image_dir, bounding_box_dir, polygon_dir, transform=None)

# 데이터 로더 생성
data_loader = DataLoader(dataset, batch_size=2, shuffle=False)

# 첫 번째 배치를 로드하여 디버깅
try:
    images, targets = next(iter(data_loader))
except FileNotFoundError as e:
    print(f"File not found: {e.filename}")
    raise
except Exception as e:
    print(f"An error occurred: {e}")
    raise

# 이미지 및 어노테이션 시각화
for i, (image, target) in enumerate(zip(images, targets)):
    # 이미지 텐서를 numpy 배열로 변환
    image = image.permute(1, 2, 0).numpy()
    
    # 마스크 시각화
    masks = target['masks'].numpy()
    combined_mask = np.sum(masks, axis=0)
    
    # 바운딩 박스 시각화
    boxes = target['boxes'].numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # 마스크 오버레이
    ax.imshow(combined_mask, alpha=0.5, cmap='jet')
    
    # 바운딩 박스 오버레이
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    
    plt.title(f"Image {i}")
    plt.show()
