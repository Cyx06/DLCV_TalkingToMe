import torch
from torchvision import models, transforms
import torch.nn as nn
from datasets import FaceDataset
import utils
import random
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm
from os.path import join
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--grid_side_length', default=4, type=int, help='num of pic in gird side length')
parser.add_argument('--pretrain', action='store_true', help='use pretrained weight')
parser.add_argument('--model_name', help='name of model to save')
parser.add_argument('--face_path', help='dir of face crop')
parser.add_argument('--seg_path', help='dir of seg files')
parser.add_argument('--epoch', default=10, type=int, help='num of epochs')


args = parser.parse_args()

random.seed(utils.seed)
# setting device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Training on', device)

# hyperparameters
lr = 1e-3
num_epochs = args.epoch
batch_size = 25

def createModel(feature_extract=False, num_classes=2):

    # set requires_grad to False to use feature extracting
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    if(args.pretrain):
        resnet = models.resnet50(weights='IMAGENET1K_V1')
    else:
        resnet = models.resnet50(weights=None)

    set_parameter_requires_grad(resnet, feature_extract)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)

    params_to_update = resnet.parameters()
    # print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in resnet.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t",name)
                continue
    else:
        for name,param in resnet.named_parameters():
            if param.requires_grad == True:
                # print("\t",name)
                continue

    return resnet, params_to_update

transform_img = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

dataset = FaceDataset(seg_dir=args.seg_path, face_dir=args.face_path, transform=transform_img, length=args.grid_side_length)
# test_dataset = FaceDataset('../data/student_data/student_data/test/seg/', '../data/test/', transform=transform_img)

train_len = int(0.9*len(dataset))
val_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, val_len])

print(len(train_set))
print(len(val_set))

train_dl = DataLoader(train_set, batch_size, num_workers=4, shuffle = True, pin_memory = True)
val_dl = DataLoader(val_set, batch_size, num_workers=4, shuffle = True, pin_memory = True)
# test_dl = DataLoader(test_dataset, batch_size, pin_memory = True)


# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
resnet, params_to_update = createModel(feature_extract=False, num_classes=2)
resnet = resnet.to(device)
# Observe that all parameters are being optimized
optimizer = torch.optim.SGD(params_to_update, lr=lr, momentum=0.9)


# calculate acc
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

best_acc = 0.5
best_loss = 10
for epoch in range(num_epochs):
    
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []

    # training
    resnet.train()
    for images, labels in tqdm(train_dl):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        out = resnet(images)                  # Generate predictions
        train_loss = F.cross_entropy(out, labels) # Calculate loss
        train_acc = accuracy(out, labels)
        train_loss.backward()
        optimizer.step()
        train_acc_list.append(train_acc.cpu().item())
        train_loss_list.append(train_loss.cpu().item())

    # validation
    resnet.eval()
    for images, labels in val_dl:
        images = images.to(device)
        labels = labels.to(device)
        out = resnet(images)                    # Generate predictions
        val_loss = F.cross_entropy(out, labels)   # Calculate loss
        val_acc = accuracy(out, labels)           # Calculate accuracy
        val_acc_list.append(val_acc.cpu().item())
        val_loss_list.append(val_loss.cpu().item())

    mean_train_acc = sum(train_acc_list)/len(train_acc_list)
    mean_val_acc = sum(val_acc_list)/len(val_acc_list)
    mean_train_loss = sum(train_loss_list)/len(train_loss_list)
    mean_val_loss = sum(val_loss_list)/len(val_loss_list)
    

    save_dir = '.'
    if mean_val_acc > best_acc:
        save_path = join(save_dir, args.model_name)
        torch.save(resnet.state_dict(), save_path)
        best_acc = mean_val_acc
    

    print('Epoch', epoch, '\nMean Train Loss',
            f'\t{mean_train_loss:.4f}',
            'Mean Train Acc', f'\t{mean_train_acc:.4f}',
            '\nMean Val Loss  ', f'\t{mean_val_loss:.4f}',
            'Mean Val Acc  ', f'\t{mean_val_acc:.4f}')
