# DreamLoc: A Deep Relevance Matching based Framework for Bug Localization

### Abstract
This repository includes the code and experimental data in our paper entitled "DreamLoc: A Deep Relevance Matching based Framework for Bug Localization". It can be used to localize bug files based on bug reports. 

### Requirements
+ python 3.7.1<br>
+ pandas 0.24.2<br>
+ gensim 3.7.2<br>
+ gitpython 3.1.1<br>
+ scikit-learn 0.20.1<br>
+ pytorch 1.3.1<br>
+ lizard 1.17.3<br>
+ numpy 1.17.4<br>
+ sent2vec<br>
+ GPU with CUDA support is also needed

### How to install
Install the dependent packages via pip:

    $ pip install pandas==0.24.2 gensim==3.7.2 GitPython==3.1.1 scikit-learn==0.20.1 lizard==1.17.3 numpy==1.17.4
    
Install pytorch according to your environment, see https://pytorch.org/.

Install Sent2vec according to the [documentation](https://github.com/epfml/sent2vec).

### How to train
1. `mkdir data`
2. download the [dataset and pre-trained models(password: 8qlg)](https://pan.baidu.com/s/1RiqcSSE2iOqOpPWSwWlGuA) into the floder `data`
3. `cd data`
4. `tar xvf data.tar`
5. `cd src`
6. run `python pipeline.py tomcat`
7. run `python dream_loc.py --project tomcat --rmm_dense_dim 100 --irff_dense_dim 20 --fusion_dense_dim 100 --k_max_pool 3 --lr 0.001`


### How to use pre-trained model
1. run `python dream_loc.py --project tomcat --rmm_dense_dim 100 --irff_dense_dim 20 --fusion_dense_dim 100 --k_max_pool 3 --just_test`
