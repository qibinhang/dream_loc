# DreamLoc: A Deep Relevance Matching based Framework for Bug Localization

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
