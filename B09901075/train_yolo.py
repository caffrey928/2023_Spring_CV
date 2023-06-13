from pathlib import Path 
from ultralytics import YOLO
import os
import sys

def create_yaml(path):
    path = os.path.join(os.getcwd(), path.replace('./', ''))
    train = os.path.join(path, 'train')
    val = os.path.join(path, 'valid')

    yaml_content = f'''
    train: {train}
    val: {val}

    names: ['Pupil']
    '''
   
    with Path('data.yaml').open('w') as f:
        f.write(yaml_content)

def train_model(device):
    model = YOLO("yolov8n-seg.pt") #可以按照需要選擇模型大小

    results = model.train(
            batch=48, #設定batchsize , 愈大愈好
            device=device, #gpu 0 
            data="data.yaml", #設定資料
            epochs=500 #訓練次數
        )

if __name__ == '__main__':
    # python3 train_yolo.py [path to dataset] [train_device]
    if(len(sys.argv) != 3 or (sys.argv[2] != '0' and sys.argv[2] != 'cpu')):
        print("Usage: python3 train_yolo.py [path to dataset] [train_device]")
        print("Train device can choose 0 or cpu")
    else:
        create_yaml(sys.argv[1])
        train_model(sys.argv[2])