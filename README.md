# Graph-Flashback_code Updating....

# Requirements
```
pip install -r requirements.txt
```

# Data Preparation

https://drive.google.com/file/d/1QXdpp0_QesJo7NZdhvoafg4MlpI_Bx-O/view?usp=sharing

**将flashback_data.zip放到根目录的data文件夹下解压后得到2个txt文件, 目录如下:**

Graph_Flashback/data/checkins-gowalla.txt

Graph_Flashback/data/checkins-4sq.txt

https://drive.google.com/file/d/1ST6GQidWVlR6yQle38MfPUSUc29t9xIT/view?usp=sharing

**再将graph_poi.zip放到根目录的KGE文件夹下解压后得到10个graph.pkl文件，目录如下：**

Graph_Flashback/KGE/gowalla_scheme2_transe_loc_temporal_10.pkl

<!-- 再将poi_graph.zip放到根目录的KGE文件夹下解压后得到36个graph.pkl文件，目录如下：

Graph_Flashback/KGE/gowalla_scheme1_transh_loc_temporal_20.pkl -->


https://drive.google.com/file/d/14l-LzoD-T3y3SAP_GU05SKAeGob6uZrX/view?usp=sharing 

下载user_loc_graph.tar

**将user_loc_graph.tar放到根目录的KGE文件夹下解压，目录如下：**

Graph_Flashback/KGE/gowalla_scheme2_transe_user-loc_50.pkl

# Model Training
进入submit文件夹，然后运行run.sh文件
```
sh run.sh
```

