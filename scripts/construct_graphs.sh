#* using pre-trained models to construct graphs

cd KGE
# TransE
## gowalla
python construct_loc_loc_graph.py --model_type transe --dataset gowalla --pretrain_model ../data/pretrained-model/gowalla_scheme2/gowalla-transe-1639638775.ckpt --version scheme2 --threshold 100 --user_count 7768 --loc_count 106994
## foursquare
python construct_loc_loc_graph.py --model_type transe --dataset foursquare --pretrain_model ../data/pretrained-model/foursquare_scheme2/foursquare-transe-1641035874.ckpt --version scheme2 --threshold 20 --user_count 45343 --loc_count 68879

## gowalla
python construct_user_loc_graph.py --model_type transe --dataset gowalla --pretrain_model ../data/pretrained-model/gowalla_scheme2/gowalla-transe-1639638775.ckpt --version scheme2 --threshold 100 --user_count 7768 --loc_count 106994
## foursquare
python construct_user_loc_graph.py --model_type transe --dataset foursquare --pretrain_model ../data/pretrained-model/foursquare_scheme2/foursquare-transe-1641035874.ckpt --version scheme2 --threshold 20 --user_count 45343 --loc_count 68879