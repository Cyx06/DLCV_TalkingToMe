# DLCV Final Project ( Talking to me )
## Group7 guardyongfuvil

```
git clone https://github.com/DLCV-Fall-2022/final-project-challenge-1-guardyongfuvil
```
# Environment

```
conda create -n group7 python=3.8

conda activate group7

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```

```
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu102.html

pip install torch-geometric

pip install pandas
```

# Download data
* Download `student_data.zip` from Kaggle and unzip it to match the directory structure shown in the SPELL section.

* SPELL
  ```
  cd final-project-challenge-1-guardyongfuvil/SPELL
  gdown "1u0bAjjc0bHj9fLohz17Qa9Dh27WZlIW_" -O ./csv_files/ttm_train.csv
  gdown "1uP3dG0Sc45dVANhnWyaXCwIfiw5Z1YNh" -O graphs.zip
  unzip graphs.zip
  mkdir graphs
  mv ./resnet18-tsm-aug_2000_0.9_cin_fsimy_normalize ./graphs/resnet18-tsm-aug_2000_0.9_cin_fsimy

  cd ..
  ```

* Image Grid (optional)
  ```shell script=
  mkdir data
  gdown 1ReUFYU4V0vLijPwaq_-MYCsloY6A7_tc -O seg.zip
  unzip seg.zip -d data/
  gdown 16AhViijYjZ_pIZ7SfDBTKl2JMsRCm11C -O face_crop.zip
  unzip face_crop.zip -d data/
  ```

# SPELL
## Directories
```
|-- SPELL
    |-- graphs
        |-- resnet18-tsm-aug_2000_0.9_cin_fsimy
            |-- train
            |-- val
            |-- test

    |-- csv_files
        |-- ttm_train.csv 
        |-- ttm_val.csv
        |-- ttm_test.csv

    |-- models
        |-- resnet18-tsm-aug_2000_0.9_cin_fsimy_lr0.001-100_c64-16_d0.2-0_s0
            |-- chckpoint_best_map.pt

|-- student_data
    |-- student_data
        |-- train
            |-- bbox
            |-- seg
        |-- test
            |-- bbox
            |-- seg

|-- dis.csv

```

## Train or Val or Test SPELL (Our Best Result on Kaggle)
```
cd SPELL
bash RunBestFinal.sh <'train' or 'val' or 'test'> <output_csv_path> <student_data_path>
```
Example:
```
bash RunBestFinal.sh train
bash RunBestFinal.sh val ./pred.csv ../student_data/student_data
bash RunBestFinal.sh test ./pred.csv ../student_data/student_data
```
* Notice that only 'test' will generate csv result file.


## Train Different GNN SPELL(optional)
```
bash RunDifferentGNNFinal.sh <'train'> <'OriginalGNN' or 'TripleLyearsGNN' or  'SingleMSU' or 'SingleUnet' or 'TripleMSU'>
```
Example:
```
bash RunDifferentGNNFinal.sh train TripleLyearsGNN
```

## Augmentation(optional)
```
python Image_Augmentation.py or Audio_Augmentation.py <input_dir> <output_dir>

Example :
python Image_Augmentation.py ./data/instance_crops_time/train ./Augmentations_images

```

# Image Grid(optional)
Train:
```shell script=
bash grid_train.sh <face crop path> <seg file path> <name of model to save>

Example :
bash grid_train.sh data/face_crop/ data/train_seg/ result.pt
```
Test:
```shell script=
bash grid_test.sh <face crop path> <seg file path> <name of saved model> <name of output file>

Example :
bash grid_test.sh data/face_crop/ data/train_seg/ model/result.pt CSV_result/result.csv
```

# Citation
```
@article{minintel,
  title={Intel Labs at ActivityNet Challenge 2022: SPELL for Long-Term Active Speaker Detection},
  author={Min, Kyle and Roy, Sourya and Tripathi, Subarna and Guha, Tanaya and Majumdar, Somdeb},
  journal={The ActivityNet Large-Scale Activity Recognition Challenge},
  year={2022},
  note={\url{https://research.google.com/ava/2022/S2_SPELL_ActivityNet_Challenge_2022.pdf}}
}
```
```
@inproceedings{min2022learning,
  title={Learning Long-Term Spatial-Temporal Graphs for Active Speaker Detection},
  author={Min, Kyle and Roy, Sourya and Tripathi, Subarna and Guha, Tanaya and Majumdar, Somdeb},
  booktitle={European Conference on Computer Vision},
  pages={371--387},
  year={2022},
  organization={Springer}
}
```


