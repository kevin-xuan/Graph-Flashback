# SIGKDD 2022. Graph-Flashback Network for Next Location Recommendation
![image](data/framework.PNG)
# Requirements
```
pip install -r requirements.txt
```

# Data Preparation

https://drive.google.com/file/d/1QXdpp0_QesJo7NZdhvoafg4MlpI_Bx-O/view?usp=sharing

**将flashback_data.zip放到根目录的data文件夹下解压后得到2个txt文件, 目录如下:**

Graph_Flashback/data/checkins-gowalla.txt

Graph_Flashback/data/checkins-4sq.txt

<!-- https://drive.google.com/file/d/1ST6GQidWVlR6yQle38MfPUSUc29t9xIT/view?usp=sharing -->
https://drive.google.com/file/d/12N9-UXPYrd4BhIlnh1B60RoV3HL5VGeJ/view?usp=sharing

**再将POI_graph.zip放到根目录的KGE文件夹下解压后得到多个graph.pkl文件，目录如下：**

Graph_Flashback/KGE/gowalla_scheme2_transe_loc_temporal_100.pkl

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
<!-- ### Citing FOGS
If you use our method as baseline, please kindly cite our IJCAI 2022 paper the using following BibTeX entry. -->
