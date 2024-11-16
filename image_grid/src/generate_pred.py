import torch
from torchvision import models, transforms
import torch.nn as nn
import utils
import random
from tqdm import tqdm
from os import listdir
from os.path import join, isfile
import pandas as pd
from PIL import Image
from random import randint
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--grid_side_length', default=4, type=int, help='num of pic in gird side length')
parser.add_argument('--file_name', help='file name of output csv')
parser.add_argument('--model_name', help='model name')
parser.add_argument('--face_path', help='dir of face crop')
parser.add_argument('--seg_path', help='dir of seg files')

args = parser.parse_args()


random.seed(utils.seed)
# setting device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print('Testing on', device)

batch_size = 32
num_classes = 2

resnet = models.resnet50(weights=None)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, num_classes)
resnet.load_state_dict(torch.load(args.model_name))
resnet.eval()

transform_img = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

seg_dir = args.seg_path
total_seg = 0
segfiles = [f for f in listdir(seg_dir) if (isfile(join(seg_dir, f)) and f[-4:] == '.csv')]
segfiles.sort()
segcounts = []
for idx, f in enumerate(segfiles):
    segfile = pd.read_csv(join(seg_dir, f))
    segcounts.append(len(segfile))
    total_seg += len(segfile)
assert len(segcounts) == len(segfiles)



result = {'id': [],
            'Predicted': []}
rows = cols = args.grid_side_length
    
for idx, f in tqdm(enumerate(segfiles)):
    segfile = pd.read_csv(join(seg_dir, f))
    for index, row in segfile.iterrows():
        start_frame = int(row['start_frame'])
        end_frame = int(row['end_frame'])

        peopleid = f[:-8]+'_'+str(row['person_id'])
        face_crop_dir = args.face_path
        face_crop_dir = join(face_crop_dir, peopleid)

        face_tomake = []
        for i in range(rows*cols):
            num_try = 0
            while True:
                img = Image.open(join(face_crop_dir, str(randint(start_frame, end_frame))+'.jpg'))
                img = img.resize((120, 120))
                px = img.load()
                if(num_try > 10):
                    img = Image.new("RGB", (120, 120), (0, 0, 0))
                    break
                if(px[60, 60] == (0, 0, 0)):
                    # print('get black image')
                    num_try += 1
                    continue
                break
            face_tomake.append(img)
        image = utils.image_grid(face_tomake, rows, cols)
        image = transform_img(image).unsqueeze(0)
        out = resnet(image)
        _, preds = torch.max(out, 1)

        resultid = peopleid+'_'+str(start_frame)+'_'+str(end_frame)
        result['id'].append(resultid)
        result['Predicted'].append(int(preds))
    
result = pd.DataFrame(result)
result.to_csv(args.file_name, index=False)

