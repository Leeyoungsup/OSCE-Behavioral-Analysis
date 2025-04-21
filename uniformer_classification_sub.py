

import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from uniformer import uniformer,uniformer_v2
from torchinfo import summary   
import torch
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Reduce
from torchvision import transforms
import json
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image   
import os
from pytorchvideo.models.hub import x3d_xs
from glob import glob
import torchvision.transforms.functional as TF
from torchvision.models.video import mvit_v2_s
device=torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(device)

key_value="필요한 물품 준비"
key_list=["손소독제", "주사기", "투약카드", "tray", "알콜솜"]
# key_value="사용한 물품 정리"
# key_list=["주사바늘 되씌우지 않음","주사바늘 손상성 폐기물 버림"]
  
params = {
    "image_size": 224,
    "frame_size": 50,
    "num_classes": len(key_list),
    "dim": (64, 128, 256, 512),
    "depth": (3, 4, 8, 3),
    "batch_size": 8,
    "mhsa_types": ('l', 'l', 'g', 'g'),
    "epoch": 200,
    "data_path": '../../data/',
    "second": '10sec',
    "class_name": key_value,
    "label_path": "../../data/label/check_list/",
    "image_channel": 3
}
params["second"]=f'{params["frame_size"]//5}sec'

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 1. 파일 리스트 생성
file_list = [f"D{str(i+1).zfill(3)}" for i in range(200)]
remove_items = ['D151', 'D159', 'D187', 'D080']
filtered_lst = [item for item in file_list if item not in remove_items]

# 2. 영상 데이터 및 라벨 저장 공간
train_images = torch.zeros(len(filtered_lst), 3, params['image_channel'], params['frame_size'],
                           params['image_size'], params['image_size'])  # [N, 3, C, T, H, W]
image_label = torch.zeros(len(filtered_lst),len(key_list))  # [N, num_classes]

# 3. 데이터 로딩
for i in tqdm(range(len(filtered_lst))):
    sample_id = filtered_lst[i]
    with open(params['label_path'] + sample_id + '.json', 'r') as f:
        check = json.load(f)

    base_path = params['data_path'] + params["second"] + '/' + params["class_name"] + '/' + sample_id
    image_list_1 = sorted(glob(base_path + '/1/*.png'))
    image_list_2 = [f.replace('/1/', '/2/') for f in image_list_1]
    image_list_3 = [f.replace('/1/', '/3/') for f in image_list_1]
    for k in range(len(key_list)):
        label = 1 if check['행동'][params["class_name"]][key_list[k]] else 0
        image_label[i,k]= label

    for j in range(params['frame_size']):
        for vid_idx, image_list in enumerate([image_list_1, image_list_2, image_list_3]):
            img = Image.open(image_list[j]).convert('RGB').resize((params['image_size'], params['image_size']))
            train_images[i, vid_idx, :, j] = trans(img)

# 4. CustomDataset 클래스 수정
class CustomDataset(Dataset):
    def __init__(self, args, video_tensor, labels, train=True):
        self.videos = video_tensor  # [N, 3, C, T, H, W]
        self.labels = labels
        self.args = args
        # 공간 증강: 랜덤 리사이즈 크롭, 랜덤 수평 뒤집기, 컬러 지터 등
        self.spatial_aug = transforms.Compose([
            transforms.RandomResizedCrop(args['image_size'], scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
        self.train = train
    def __getitem__(self, idx):
        video1 = self.videos[idx, 0]
        video2 = self.videos[idx, 1]
        video3 = self.videos[idx, 2]
        label = self.labels[idx]
        return video1, video2, video3, label

    def __len__(self):
        return len(self.videos)


# 5. 학습/테스트 분할
split = int(len(train_images) * 0.7)
train_dataset = CustomDataset(params, train_images[:split], image_label[:split].float(), train=True)
test_dataset  = CustomDataset(params, train_images[split:], image_label[split:].float(), train=False)

# 6. DataLoader 구성
train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True)
test_dataloader  = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, drop_last=True)


class Multix3d(nn.Module):
    def __init__(self, num_classes=len(key_list), pretrained=True):
        super().__init__()

        # 세 개의 MViTv2-S 모델을 생성 (출력 차원은 768)
        self.backbone1 = x3d_xs(pretrained=pretrained, progress=True)
        self.backbone2 =  x3d_xs(pretrained=pretrained, progress=True)
        self.backbone3 = x3d_xs(pretrained=pretrained, progress=True)
        self.in_features = self.backbone1.blocks[-1].proj.in_features
        # 세 feature를 concat 후 최종 분류
        self.classifier = nn.Sequential(
            nn.Linear(400* 3, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, video1, video2, video3):
        # 입력: [B, C, T, H, W]
        feat1 = self.backbone1(video1)  # [B, 768]
        feat2 = self.backbone2(video2)
        feat3 = self.backbone3(video3)
    
        fused = torch.cat([feat1, feat2, feat3], dim=1)  # [B, 2304]
        return self.classifier(fused)  # [B, num_classes]
    
model = Multix3d(
    num_classes=params['num_classes']
).to(device)

# 입력 비디오 크기 정의
video_size = (
    params['batch_size'],       # B
    params['image_channel'],    # C = 3
    params["frame_size"],                         # T = 16 (MViT 기준)
    params['image_size'],       # H
    params['image_size']        # W
)

# Optimizer & Loss 정의
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

# 디렉토리 생성 함수
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 모델 구조 출력
summary(
    model,
    input_size=[video_size, video_size, video_size],  # 3개 비디오 입력
    device=str(device)
)

best_val_loss = float('inf')

for epc in range(params['epoch']):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    steps = 0

    with tqdm(train_dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for video1, video2, video3, lab in tqdmDataLoader:
            optimizer.zero_grad()

            # 데이터 준비
            video1 = video1.to(device)
            video2 = video2.to(device)
            video3 = video3.to(device)
            lab = lab.to(device)

            # 모델 forward
            output = model(video1, video2, video3) # (B, num_classes)
            loss = criterion(output, lab)  # CrossEntropyLoss expects class index

            # 역전파
            loss.backward()
            optimizer.step()

            # 통계
            total_loss += loss.item()
            steps += 1

            # Accuracy 계산
            preds = torch.where(output>0.5,1,0)
            targets = lab
            total_correct += (preds == targets).sum().item()/params['num_classes']
            total_samples += lab.size(0)

            tqdmDataLoader.set_postfix(
                ordered_dict={
                    "epoch": epc + 1,
                    "loss": total_loss / steps,
                    "acc": f"{100 * total_correct / total_samples:.2f}%",
                    "batch": lab.size(0),
                    "LR": optimizer.param_groups[0]["lr"]
                }
            )

    # ======== Validation ========
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_steps = 0

    with torch.no_grad():
        with tqdm(test_dataloader, desc=f"[Valid] Epoch {epc+1}", dynamic_ncols=True) as tqdmVal:
            for video1, video2, video3, lab in tqdmVal:
                video1 = video1.to(device)
                video2 = video2.to(device)
                video3 = video3.to(device)
                lab = lab.to(device)

                output = model(video1, video2, video3)
                loss = criterion(output, lab)

                val_loss += loss.item()
                val_steps += 1
                preds = torch.where(output>0.5,1,0)
                targets = lab
                val_correct += (preds == targets).sum().item()/params['num_classes']
                val_total += lab.size(0)

                tqdmVal.set_postfix(
                    val_loss=f"{val_loss / val_steps:.4f}",
                    val_acc=f"{100 * val_correct / val_total:.2f}%"
                )

    # ======== 모델 저장 ========
    avg_val_loss = val_loss / val_steps
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        c1 = params["class_name"]
        c2 = params["second"]
        create_dir(f"../../model/{c1}/")
        torch.save(model.state_dict(), f"../../model/{c1}/best_model_{c2}.pt")
        print(f"✅ Model saved at epoch {epc+1} with validation loss {best_val_loss:.4f}")
