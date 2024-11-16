import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from os import listdir
from os.path import join, isfile
from random import randint
import utils

class FaceDataset(Dataset):
    def __init__(self, seg_dir, face_dir, transform=None, target_transform=None, length=2):
        self.seg_dir = seg_dir
        self.face_dir = face_dir
        self.transform = transform
        self.target_transform = target_transform
        self.length = length

        total_seg = 0       
        segfiles = [f for f in listdir(self.seg_dir) if (isfile(join(self.seg_dir, f)) and f[-4:] == '.csv')]
        segfiles.sort()
        segcounts = []
        for idx, f in enumerate(segfiles):
            # print(idx, f)
            if(f[-4:] != '.csv'):
                continue
            segfile = pd.read_csv(join(self.seg_dir, f))
            segcounts.append(len(segfile))
            total_seg += len(segfile)
            # if(len(segfile) == 0):
            #     print(f)
        
        assert len(segcounts) == len(segfiles)
        self.segfiles = segfiles
        self.segcounts = segcounts
        # print(segcounts)
        self.total_seg = total_seg

    def __len__(self):
        return self.total_seg

    def __getitem__(self, idx):
        fileidx = 0
        for i, count in enumerate(self.segcounts):
            # print(idx)
            if(idx >= count):
                idx -= count
            else:
                fileidx = i
                break 
        
        
        file = pd.read_csv(join(self.seg_dir, self.segfiles[fileidx]))
        try:
            label = int(file.iloc[idx]['ttm'])
        except:
            label = 0

        # print('person', file.iloc[idx]['person_id'], type(str(file.iloc[idx]['person_id'])))

        face_crop_dir = self.face_dir
        peopleid = self.segfiles[fileidx][:-8]+'_'+str(file.iloc[idx]['person_id'])
        # print('peopleid', peopleid)
        face_crop_dir = join(face_crop_dir, peopleid)

        start_frame = int(file.iloc[idx]['start_frame'])
        end_frame = int(file.iloc[idx]['end_frame'])
        
        rows = cols = self.length
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
        # image.show()
        # exit()

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label