import os

import torch





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
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        lr_scheduler.step()
        print(f"Epoch {epoch}/{num_epochs}, Loss: {losses.item()}")
        
    return model


def main():
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    if not os.path.exists(opt.data_path):
        os.makedirs(opt.data_path)
    
    dataloader = init(opt.data_path)
    model = get_model(num_classes=91)  # 예시: COCO 데이터셋의 91개 클래스
    model = train_model(model, dataloader, num_epochs=10)  # 에폭 수 조정

if __name__ == "__main__":
    main()
