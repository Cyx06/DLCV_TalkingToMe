from scipy.io.wavfile import read, write
import os
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='SPELL')
parser.add_argument('--input', type=str, default='./data/instance_wavs_time/train', help='which gpu to run the train_val')
parser.add_argument('--output', type=str, default='./Augmentations_audios', help='name of the features')

args = parser.parse_args()

# change here for input idr
audios_dir = args.input
audios_file = os.listdir(audios_dir)
# print(audios_file)
for i in range(len(audios_file)):
    print(audios_file[i])
    Hz, original_data = read(os.path.join(audios_dir, audios_file[i]))
    temps = []
    counter = 0
    while 1:
        random_number = random.randrange(len(audios_file))
        Hz, temp = read(os.path.join(audios_dir, audios_file[random_number]))
        temps.append(list(temp))
        counter += len(temp)
        if counter >= len(original_data): break
    noise_data = []
    for temp in temps:
        noise_data += temp
    noise_data = np.array(noise_data[:len(original_data)])
    random_number = random.uniform(0.9, 1.1)
    result_data = (noise_data * 0.2 + original_data * 0.8) * random_number
    # print(result_data)
    # change here for result idr
    write(os.path.join(args.output, audios_file[i][:len(audios_file[i]) - 4] + "_aug_ed" + ".wav"), Hz, result_data)
