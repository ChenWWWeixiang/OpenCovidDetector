# Development and Evaluation of an AI System for COVID-19 Diagnosis
This project hosts the inference code for implementing for covid-19 diagnosis system, as presented in our paper: Development and Evaluation of an AI System for COVID-19 Diagnosis (pre-print paper is still under screening)

Environment
-------
python==3.6.1
matplotlib==3.1.2
six==1.13.0
torch==1.3.1
scikit_image==0.16.2
imageio==2.6.1
scipy==1.3.3
numpy==1.15.3
opencv_python==4.1.1.26
pandas==0.23.4
torchvision==0.4.2
Pillow==7.0.0
pydicom==1.4.2
pyradiomics
scikit_learn==0.22.2.post1
seaborn==0.10.0
SimpleITK==1.2.4
skimage
tensorboardX==2.0
toml==0.10.0
xlrd==1.2.0



Data Preparation
-------
1. Resampling: python data/resample.py
2. Cut into jpg and normalize gray scale: python test_hu.py
3. Split dataset: python data/get_set_seperate_jpg.py
4. Lung segmentation mask: using Deeplabv1 (https://github.com/DrSleep/tensorflow-deeplab-resnet)

Train and Test
-------
Our model train and validate on slice level, while test in volume level. Parameters in options_lip.toml should be changed firstly.
train: python main.py
test: python testengine.py

Abmormal Slice Locating
-------
fine-tune using main.py
test in multi_period_scores/

Radiomics and LASSO analysis
-------
1. Extract features: python get_r_features.py
2. LASSO analysis: python plot_lasso_mse.py

Fractal Dimension
-------
in fractal-dimension/