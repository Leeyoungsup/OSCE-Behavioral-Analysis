
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from uniformer import uniformer
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
from glob import glob
device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(device)

key_list=['물과 비누로 손위생',
 '투약처방과 투약원칙 확인 (손으로 짚어서)',
 '근육주사 약물을 정확한 용량과 방법으로 준비',
 '손소독제로 손위생',
 '대상자의 입원팔찌와 투약카드 대조하여 확인',
 '주사부위 노출 후, 주사부위 선정(삼각근)',
 '물과 비누,알콜젤로 손위생 수행',
 '소독솜으로 닦고, 한손으로 주사바늘 뚜껑 제거',
 '주사바늘 90도로 주사부위 찌름',
 '내관당겨보고, 약물 천천히 주입',
 '삽입각도와 같이 빼고, 주사부위 압박',
 '환의 정리',
 '물과 비누로 손위생 (종료 후)']

params={
    "image_size": 224,
    "frame_size": 50,
    "num_classes": 2,
    "dim": (64, 128, 256, 512),
    "depth": (3, 4, 8, 3),
    "batch_size": 2,
    "mhsa_types": ('l', 'l', 'g', 'g'),
    "epoch": 200,
    "data_path": '../../data/',
    "second": '10sec',
    "class_name": key_list[3],
    "label_path": "../../data/label/check_list/",
    "image_channel": 3
}



params["second"]=f'{params["frame_size"]//5}sec'


file_list=[f"D{str(i+1).zfill(3)}" for i in range(200)]
remove_items = ['D151', 'D159', 'D187', 'D080']
filtered_lst = [item for item in file_list if item not in remove_items]
trans = transforms.Compose([
    transforms.ToTensor(),
])

class CustomDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, parmas, video, label):

        self.images = video
        self.args = parmas
        self.label = label

    def __getitem__(self, index):
        video1 = self.images[index,0]
        video2 = self.images[index,1]
        video3 = self.images[index,2]
        label = self.label[index]

        return video1,video2,video3, label

    def __len__(self):
        return len(self.images)



image_label = []
train_images = torch.zeros(len(filtered_lst),3,params['image_channel'],params['frame_size'],params['image_size'],params['image_size'])
for i in tqdm(range(len(filtered_lst))):
    data_path=params['data_path']+filtered_lst[i]+'/*.png'
    with open(params['label_path']+filtered_lst[i]+'.json', 'r') as f:
        check = json.load(f)
    image_list_1 = glob(params['data_path']+params["second"]+'/'+params["class_name"]+'/'+filtered_lst[i]+'/1/*.png')
    image_list_1.sort()
    image_list_2 =[f.replace('/1/', '/2/') for f in image_list_1]
    image_list_3 =[f.replace('/1/', '/3/') for f in image_list_1]
    if check['행동'][params["class_name"]]==True:
        image_label.append(1)
    else:
        image_label.append(0)
    for j in range(params['frame_size']):
        train_images[i,0,:,j]=trans(Image.open(image_list_1[j]).convert('RGB').resize((params['image_size'], params['image_size'])))
        train_images[i,1,:,j]=trans(Image.open(image_list_2[j]).convert('RGB').resize((params['image_size'], params['image_size'])))
        train_images[i,2,:,j]=trans(Image.open(image_list_3[j]).convert('RGB').resize((params['image_size'], params['image_size'])))

train_dataset = CustomDataset(
    params, train_images[:-30], F.one_hot(torch.tensor(image_label[:-30])))
test_dataset = CustomDataset(
    params, train_images[-30:], F.one_hot(torch.tensor(image_label[-30:])))
train_dataloader = DataLoader(
    train_dataset, batch_size=params['batch_size'], shuffle=True,drop_last=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=params['batch_size'], shuffle=True,drop_last=True)

model = uniformer.MultiVideoUniformer(
    num_classes = params['num_classes'],                 # number of output classes
    dims = params['dim'],         # feature dimensions per stage (4 stages)
    depths = params['depth'],              # depth at each stage
    mhsa_types = params['mhsa_types']   # aggregation type at each stage, 'l' stands for local, 'g' stands for global
).to(device)

video_size = (params['batch_size'], params['image_channel'], params['frame_size'], params['image_size'], params['image_size']) # (batch, channels, time, height, width)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
summary(
    model,
    input_size=[
        (params['batch_size'], params['image_channel'], params['frame_size'], params['image_size'], params['image_size']),  # video1
        (params['batch_size'], params['image_channel'], params['frame_size'], params['image_size'], params['image_size']),  # video2
        (params['batch_size'], params['image_channel'], params['frame_size'], params['image_size'], params['image_size'])   # video3
    ],
    device=device
)
def create_dir(path):  
    if not os.path.exists(path):
        os.makedirs(path)
        
        
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
            output = model(video1, video2, video3)  # (B, num_classes)
            loss = criterion(output, lab.argmax(dim=1))  # CrossEntropyLoss expects class index

            # 역전파
            loss.backward()
            optimizer.step()

            # 통계
            total_loss += loss.item()
            steps += 1

            # Accuracy 계산
            preds = output.argmax(dim=1)
            targets = lab.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
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
                loss = criterion(output, lab.argmax(dim=1))

                val_loss += loss.item()
                val_steps += 1
                preds = output.argmax(dim=1)
                targets = lab.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += lab.size(0)

                tqdmVal.set_postfix(
                    val_loss=f"{val_loss / val_steps:.4f}",
                    val_acc=f"{100 * val_correct / val_total:.2f}%"
                )

    # ======== 모델 저장 ========
    avg_val_loss = val_loss / val_steps
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        c1=params["class_name"]
        c2=params["second"]
        create_dir(f"../../model/{c1}/")
        torch.save(model.state_dict(), f"../../model/{c1}/best_model_{c2}.pt")
        print(f"✅ Model saved at epoch {epc+1} with validation loss {best_val_loss:.4f}")