# new_Flashback_code

# Requirements
* torch 1.10

* numpy

* pickle

* tqdm

# Data Preparation

将flashback_data.zip放到根目录的data文件夹下解压后得到2个txt文件, 目录如下:

new_Flashback_code/data/checkins-gowalla.txt

new_Flashback_code/data/checkins-4sq.txt


再将poi_graph.zip放到根目录的KGE文件夹下解压后得到36个graph.pkl文件，目录如下：

new_Flashback_code/KGE/gowalla_scheme1_transh_loc_temporal_20.pkl


# Model Training
进入submit文件夹，然后运行run.sh文件
```
sh run.sh
```


