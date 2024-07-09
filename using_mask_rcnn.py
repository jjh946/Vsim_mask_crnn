import argparse
import os
import numpy as np
import torch

import cv2
import torchvision
import colorsys
import random
import natsort
from torch.utils.data import DataLoader
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as transforms


print("torch vision version:" + torchvision.__version__)
print("torch version:" + torch.__version__)
print("torch cuda version:" + torch.version.cuda)
from datasets import ImageDataset

# 명령행 인자 설정
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='배치 크기')
parser.add_argument('--root', type=str, default='./', help='루트 디렉토리')
parser.add_argument('--data_path', type=str, default='./images/', help='데이터셋 디렉토리')
parser.add_argument('--save_path', type=str, default='./output/', help='결과 저장 디렉토리')
parser.add_argument('--cuda', action='store_false', help='GPU 사용 여부')
parser.add_argument('--n_cpu', type=int, default=8, help='데이터 로딩에 사용할 CPU 스레드 수')
parser.add_argument('--input_nc', type=int, default=3, help='입력 데이터 채널 수')
opt = parser.parse_args()

# COCO 클래스 이름
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

#######################################
# 모델 정의
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

#####################################################
# 모델 테스트 모드 설정
model.eval()

# 데이터 로더 초기화
def init(data_path):
    # 데이터셋 로더
    transforms_ = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = DataLoader(ImageDataset(data_path, transforms_=transforms_),
                            batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
    return dataloader

# 마스크 적용 함수
def apply_mask(image, mask, labels, boxes, file_name, save_path):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    alpha = 1
    beta = 0.6  # 세그멘테이션 맵의 투명도
    gamma = 0  # 각 합에 추가되는 스칼라
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
    channel = image.shape[2]
    _, _, w, h = mask.shape
    segmentation_map = np.zeros((w, h, 3), np.uint8)
    
    for n in range(mask.shape[0]):
        if labels[n] == 0:
            continue
        color = COLORS[random.randrange(0, len(COLORS))]
        segmentation_map[:, :, 0] = np.where(mask[n, 0] > 0.5, COLORS[labels[n]][0], segmentation_map[:, :, 0])
        segmentation_map[:, :, 1] = np.where(mask[n, 0] > 0.5, COLORS[labels[n]][1], segmentation_map[:, :, 1])
        segmentation_map[:, :, 2] = np.where(mask[n, 0] > 0.5, COLORS[labels[n]][2], segmentation_map[:, :, 2])
        image = cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, dtype=cv2.CV_8U)
        cv2.rectangle(image, boxes[n][0], boxes[n][1], color=color, thickness=2)
        cv2.putText(image, class_names[labels[n]], (boxes[n][0][0], boxes[n][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2, lineType=cv2.LINE_AA)
    
    cv2.imwrite(save_path + 'seg_' + file_name, image)

# 랜덤 색상 생성
def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

# 텐서를 이미지로 변환
def tensor2im(tensor):
    tensor = 127.5 * (tensor[0].data.float().numpy() + 1.0)
    img = tensor.astype(np.uint8)
    img = np.transpose(tensor, (1, 2, 0))
    return img

# 마스크 R-CNN 실행
def mask_rcnn(file_list, dataloader, save_path):
    for i, batch in enumerate(dataloader):
        img = batch['A']
        print(file_list[i])
        result = model(img)
        image = tensor2im(img)
        scores = list(result[0]['scores'].detach().numpy())
        thresholded_preds_inidices = [scores.index(i) for i in scores if i > 0.965]
        thresholded_preds_count = len(thresholded_preds_inidices)
        mask = result[0]['masks']
        mask = mask[:thresholded_preds_count]
        labels = result[0]['labels']
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in result[0]['boxes']]
        boxes = boxes[:thresholded_preds_count]
        
        mask = mask.data.float().numpy()
        
        apply_mask(image, mask, labels, boxes, file_list[i], save_path)

# 메인 함수
def main():
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    if not os.path.exists(opt.data_path):
        os.makedirs(opt.data_path)
    file_list = os.listdir(opt.data_path)
    file_list = natsort.natsorted(file_list)
    dataloader = init(opt.data_path)
    mask_rcnn(file_list, dataloader, opt.save_path)

if __name__ == "__main__":
    main()
