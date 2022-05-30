# Hand Signs Classification Competition from DACON
I stand in 39th out of 150 persons.

Competition link is https://dacon.io/competitions/official/235896/overview/description

The Mission was Classifying hand sign images that express 0 to 10 seperately.

![image](https://user-images.githubusercontent.com/104220612/170996814-7c62859d-065c-4d1f-8605-fb0ff380dd59.png)

# Key Idea
I used **'Ensemble Train'**, **'Image augmentaiton'**

Actually the number of train images was about 860, so it was necessary to augmentate images.

# Frame works
**Pytorch** for ensemble train.

**Albumentations** for image augmentation.

# Code example
```python
# Load efficientnet_b3 model

loaded_model = torch.load('/content/drive/MyDrive/DACON_Image/weights/b3_model.pt')
model_b3 = Network_b3().to(device)
model_b3.load_state_dict(loaded_model['model_state_dict'])

# Load wide_resnet50_2 model

loaded_model = torch.load('/content/drive/MyDrive/DACON_Image/weights/wrn_model.pt')
model_wrn = Network_wrn().to(device)
model_wrn.load_state_dict(loaded_model['model_state_dict'])
```

# How to use
1. Download the images from DACON.
2. Download Ensemble_Train.ipynb.
3. Take them in to same colab directory.
4. Activate Ensemble_Train.ipynb and change the directory path to your own path
5. Run the code
