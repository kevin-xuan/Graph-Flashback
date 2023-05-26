# SIGKDD 2022. Graph-Flashback Network for Next Location Recommendation
![image](data/framework.PNG)
# Requirements
```
pip install -r requirements.txt
```

# Data Preparation

**1. place [flashback_data.zip](https://drive.google.com/file/d/1QXdpp0_QesJo7NZdhvoafg4MlpI_Bx-O/view?usp=sharing) into Graph_Flashback/data/ and unzip the file as following:**

Graph_Flashback/data/checkins-gowalla.txt

Graph_Flashback/data/checkins-4sq.txt

<!-- https://drive.google.com/file/d/1ST6GQidWVlR6yQle38MfPUSUc29t9xIT/view?usp=sharing -->

**2. place [POI_graph.zip](https://drive.google.com/file/d/12N9-UXPYrd4BhIlnh1B60RoV3HL5VGeJ/view?usp=sharing) into Graph_Flashback/KGE/ and unzip the file as following :**

Graph_Flashback/KGE/gowalla_scheme2_transe_loc_temporal_100.pkl

**3. Moreover, the [pre-trained models](https://drive.google.com/file/d/1oUXQjtnDrUnmdhVSGY64WrXBngzD1qhz/view?usp=sharing) are also provided. (If you are interested in the code of pre-trained model, you can find the download link [here](https://github.com/kevin-xuan/Graph-Flashback/issues/1#issuecomment-1235372011))** 

**4. The friend datasets have been uploaded into the [data file](https://github.com/kevin-xuan/Graph-Flashback/tree/main/data). You can also download the user profile file by the link presented in Section 6 "EXPERIMENTS" in our paper.**

<!-- 再将poi_graph.zip放到根目录的KGE文件夹下解压后得到36个graph.pkl文件，目录如下：

Graph_Flashback/KGE/gowalla_scheme1_transh_loc_temporal_20.pkl -->


<!--https://drive.google.com/file/d/14l-LzoD-T3y3SAP_GU05SKAeGob6uZrX/view?usp=sharing 

下载user_loc_graph.tar

**将user_loc_graph.tar放到根目录的KGE文件夹下解压，目录如下：**

Graph_Flashback/KGE/gowalla_scheme2_transe_user-loc_50.pkl-->

# Model Training

Gowalla
```
python train.py --trans_loc_file KGE/gowalla_scheme2_transe_loc_temporal_100.pkl --trans_interact_file KGE/gowalla_scheme2_transe_user-loc_100.pkl
```

Foursquare
```
python train.py --dataset checkins-4sq.txt --trans_loc_file KGE/foursquare_scheme2_transe_loc_temporal_20.pkl --trans_interact_file KGE/foursquare_scheme2_transe_user-loc_20.pkl
```
# Constructing new datasets
In order to train your model on your own datasets, you should run [generate_triplet.py](https://github.com/kevin-xuan/Graph-Flashback/blob/380d7bf98f7996d72ab126e1339d5078b8b5a7e3/KGE/generate_triplet.py#L237) by modifying some default settings such as dataset_file and [refine.py](https://github.com/kevin-xuan/Graph-Flashback/blob/380d7bf98f7996d72ab126e1339d5078b8b5a7e3/KGE/refine.py#L5) to extract triplets files.
Moreover, we provide the **KGE** code by the [link](https://github.com/kevin-xuan/Graph-Flashback/issues/1#issuecomment-1235372011), and you should move the extracted files into the corresponding dir, e.g., './data/gowalla/'. Finally, using your pre-trained model to construct the location graph by the [construct_loc_loc_graph.py](https://github.com/kevin-xuan/Graph-Flashback/blob/3d5d42bbd50e39d797564a3aa880232ffcaccdb5/KGE/construct_loc_loc_graph.py#L162).

# Citing
If you use Graph-Flashback in your research, please cite the following [paper](https://dl.acm.org/doi/10.1145/3534678.3539383):
```
@inproceedings{DBLP:conf/kdd/RaoCLSYH22,
  author    = {Xuan Rao and
               Lisi Chen and
               Yong Liu and
               Shuo Shang and
               Bin Yao and
               Peng Han},
  title     = {Graph-Flashback Network for Next Location Recommendation},
  booktitle = {{KDD} '22: The 28th {ACM} {SIGKDD} Conference on Knowledge Discovery
               and Data Mining, Washington, DC, USA, August 14 - 18, 2022},
  year      = {2022}
}
```
