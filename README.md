# retrieval.pytorch

## dependence

* python3
* pytorch
* numpy
* scikit-learn
* tqdm
* yacs

You can install `yacs` by `pip`. Other dependencies can be installed by 'conda'.

## prepare dataset

first, you need to download the dataset from [here](https://developer.huawei.com/consumer/en/activity/devStarAI/algo/competition.html#/preliminary/info/digix-trail-04/data). Then, you can move them into the directory `$DATASET` and decompress them by
```
unzip train_data.zip
unzip test_data_A.zip
unzip test_data_B.zip
```

Then remove the empty directory in `train_data`:
```
cd train_data
rm -rf DIGIX_001453
rm -rf DIGIX_001639
rm -rf DIGIX_002284
```

Finally, you need to edit the file `src/dataset/datasets.py` and set the correct values for `traindir`, `test_A_dir`, `test_B_dir`.
```
traindir = '$DATASET/train_data'
test_A_dir = '$DATASET/test_data_A'
test_B_dir = '$DATASET/test_data_B'
```

## Train the network to extract feature

You can train `dla102x` and `resnet101` by the below comands.
```
python experiments/DIGIX/dla102x/cgd_margin_loss.py
python experiments/DIGIX/resnet101/cgd_margin_loss.py
```

To train `fishnet99`, `hrnet_w18` and `hrnet_w30`, you need to download their imagenet pretrained weights from [here](https://box.nju.edu.cn/d/bc9ded1409f341fc8ae7/). Specifically, download `fishnet99_ckpt.tar` for `fishnet99`, download `hrnetv2_w18_imagenet_pretrained.pth` for `hrnet_w18`, download `hrnetv2_w30_imagenet_pretrained.pth` for `hrnet_w30`. Then you need to move these weights to `~/.cache/torch/hub/checkpoints` to make sure `torch.hub.load_state_dict_from_url` can find them.

Then, you can train `fishnet99`, `hrnet_w18`, `hrnet_w30` by
```
python experiments/DIGIX/fishnet99/cgd_margin_loss.py
python experiments/DIGIX/hrnet_w18/cgd_margin_loss.py
python experiments/DIGIX/hrnet_w30/cgd_margin_loss.py
```

After Training, the model weights can be found in `results/DIGIX/{model}/cgd_margin_loss/{time}/transient/checkpoint.final.ckpt`. We also provide these [weights file](https://box.nju.edu.cn/d/bc9ded1409f341fc8ae7/).

## extract features for retrieval

You can download the pretrained model from [here](https://box.nju.edu.cn/d/bc9ded1409f341fc8ae7/) and move them to `pretrained` directory.

Then, run the below comands.
```
python experiments/DIGIX_test_B/dla102x/cgd_margin_loss_test_B.py
python experiments/DIGIX_test_B/resnet101/cgd_margin_loss_test_B.py
python experiments/DIGIX_test_B/fishnet99/cgd_margin_loss_test_B.py
python experiments/DIGIX_test_B/hrnet_w18/cgd_margin_loss_test_B.py
python experiments/DIGIX_test_B/hrnet_w30/cgd_margin_loss_test_B.py
```

When finished, the query feature for `test_data_B` can be found in `results/DIGIX_test_B/{model}/cgd_margin_loss_test_B/{time}/query_feat`. And the gallery feature can be found in `results/DIGIX_test_B/{model}/cgd_margin_loss_test_B/{time}/gallery_feat`. 

## Post process

You can download features from [here](https://box.nju.edu.cn/d/c70b51aed2de4277b5d3/). Then, you can put it into the directory `features` and decompress the files by
```
tar -xvf DIGIX_test_B_dla102x_5088.tar
tar -xvf DIGIX_test_B_fishnet99_5153.tar
tar -xvf DIGIX_test_B_hrnet_w18_5253.tar
tar -xvf DIGIX_test_B_hrnet_w30_5308.tar
tar -xvf DIGIX_test_B_resnet101_5059.tar
```

Then the `features` directory will be organized like this:
```
|-- DIGIX_test_B_dla102x_5088.tar  
|-- DIGIX_test_B_fishnet99_5153.tar  
|-- DIGIX_test_B_hrnet_w18_5253.tar  
|-- DIGIX_test_B_hrnet_w30_5308.tar  
|-- DIGIX_test_B_resnet101_5059.tar 
|-- DIGIX_test_B_dla102x_5088  
| |-- gallery_feat  
| |-- query_feat  
|-- DIGIX_test_B_fishnet99_5153  
| |-- gallery_feat  
| |-- query_feat  
|-- DIGIX_test_B_hrnet_w18_5253  
| |-- gallery_feat  
| |-- query_feat  
|-- DIGIX_test_B_hrnet_w30_5308  
| |-- gallery_feat  
| |-- query_feat  
|-- DIGIX_test_B_resnet101_5059  
| |-- gallery_feat  
| |-- query_feat  
```

Now, post process can be executed by
```
python post_process/rank.py --gpu 0 features/DIGIX_test_B_fishnet99_5153 features/DIGIX_test_B_dla102x_5088 features/DIGIX_test_B_hrnet_w18_5253 features/DIGIX_test_B_hrnet_w30_5308 features/DIGIX_test_B_resnet101_5059
```

