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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms

print("torch vision version:" + torchvision.__version__)
print("torch version:" + torch.__version__)
print("torch cuda version:" + torch.version.cuda)
from datasets import CustomDataset

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

# 모델 정의
def get_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

# 모델 학습 함수
def train_model(model, dataloader, num_epochs):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            images = list(image.to(device) for image in batch['A'])
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch['target']]
            print("Batch target structure:", targets)  # 디버깅 출력 추가
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        lr_scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")
        
    return model

# 데이터 로더 초기화
def init(data_path):
    transforms_ = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataset = CustomDataset(data_path, transforms_=transforms_)
    dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
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

# 텐서를 이미지로 변환
def tensor2im(tensor):
    tensor = 127.5 * (tensor[0].data.float().numpy() + 1.0)
    img = tensor.astype(np.uint8)
    img = np.transpose(tensor, (1, 2, 0))
    return img

# 마스크 R-CNN 실행
def mask_rcnn(file_list, dataloader, save_path, model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    
    for i, batch in enumerate(dataloader):
        img = batch['A'].to(device)
        print(file_list[i])
        with torch.no_grad():
            result = model([img])
        
        image = tensor2im(img.cpu())
        scores = list(result[0]['scores'].cpu().numpy())
        thresholded_preds_inidices = [scores.index(i) for i in scores if i > 0.965]
        thresholded_preds_count = len(thresholded_preds_inidices)
        mask = result[0]['masks'][:thresholded_preds_count]
        labels = result[0]['labels'][:thresholded_preds_count].cpu().numpy()
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in result[0]['boxes'][:thresholded_preds_count].cpu().numpy()]
        
        apply_mask(image, mask.cpu().numpy(), labels, boxes, file_list[i], save_path)

# 메인 함수
def main():
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    if not os.path.exists(opt.data_path):
        os.makedirs(opt.data_path)
    file_list = os.listdir(opt.data_path)
    file_list = natsort.natsorted(file_list)
    
    dataloader = init(opt.data_path)
    model = get_model(num_classes=len(class_names))  # 예시: COCO 데이터셋의 클래스 수 사용
    model = train_model(model, dataloader, num_epochs=10)  # 에폭 수 조정
    
    # Save the model
    torch.save(model.state_dict(), os.path.join(opt.save_path, 'model.pth'))
    
    mask_rcnn(file_list, dataloader, opt.save_path, model)

if __name__ == "__main__":
    main()