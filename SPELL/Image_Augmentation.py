import os
from PIL import Image
from torchvision import transforms
import random
import argparse

parser = argparse.ArgumentParser(description='SPELL')
parser.add_argument('--input', type=str, default='./data/instance_crops_time/train', help='which gpu to run the train_val')
parser.add_argument('--output', type=str, default='./Augmentations_images', help='name of the features')

args = parser.parse_args()


# here for change input and output
input_dir = args.input
output_dir = args.output
folders = os.listdir(input_dir)

for folder in folders:
    print(folder)
    faces = os.listdir(os.path.join(input_dir, folder))
    for face in faces:
        img = Image.open(os.path.join(input_dir, folder, face))
        # print(img.size)
        size = img.size
        random_dsize = [random.randint(0, int(size[0] * 0.2)), random.randint(0, int(size[1] * 0.2))]
        # print(random_dsize)
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomRotation(8),
            transforms.RandomCrop(size=(size[1] - random_dsize[1], size[0] - random_dsize[0]))
        ])

        img = transform(img)
        os.makedirs(os.path.join(output_dir, folder + "_aug_ed"),exist_ok=True)
        img.save(os.path.join(output_dir, folder + "_aug_ed", face))