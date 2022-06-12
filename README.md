# DeepConvNet-master

This repo contains the code and the pre-trained models used in the paper:"Shallow and Deep Convolutional Networks for Saliency Prediction".

## Train a model

Our training code is based on the [SALICON](http://salicon.net/challenge-2017/) dataset, we assume you already download and unzip the images and groundtruths under your workspace.

```
SaliconDataset
└───train
│     │   *.jpg
|     |
└───val
|     |  *.jpg
|     |
└───maps
      └───train
      │     │   *.png
      |     |
      └───val
      │     │   *.png      
```

After making sure that the path of the SALICON dataset is correct, please save the path files "**trainList.txt**" and "**valList.txt**" under folder "SaliconDataset/".

```
SaliconDataset
└───trainList.txt
└───valList.txt
```

Our training code "**main.py**" contains a pre-trained model [VGG-16](http://data.lip6.fr/cadene/pretrainedmodels/vggm-786f2434.pth). Please save the model under folder "Models/".

The trained models will be saved under folder "Models/Model_Save_new/"

## Make a prediction

Note that our pre-trained model "**DeepConvNet.pth**" ([google drive](https://drive.google.com/file/d/1ZBZaqGe3LTmwtTJR03JpoNpnuzYPlcyF/view?usp=sharing)) and the VGG-16 pre-trained model are not included. 

Our prediction code "**test.py**" assumes that the test set is SALICON test set. You are supposed to download and unzip the images under the folder "SaliconDataset/test/". 

```
SaliconDataset
└───test
│     │   *.jpg
```

Besides, please save the path file "**testList.txt**" under the folder "SaliconDataset/".

Before running our code, make sure the pre-trained model is saved under folder "Models/Model_Save_new/". Please check if the pre-trained model loaded in the code "**test.py**" exists. 

It will save the prediction under folder "result/test/", you might want to change the path and the prediction file name.

------

Our prediction code "**test_mit.py**" assumes that the test set is [MIT300](http://saliency.mit.edu/BenchmarkIMAGES.zip) test set. You are supposed to download and unzip the images under the folder "SaliconDataset/MITtest/". 

```
SaliconDataset
└───MITtest
│     │   *.jpg
```

Besides, please save the path file "**testMIT.txt**" under the folder "SaliconDataset/" and check if the pre-trained model loaded in the code "**test_mit.py**" exists. 

It will save the prediction under folder "result/mit/", you might want to change the path and the prediction file name.

