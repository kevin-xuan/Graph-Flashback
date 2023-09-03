# gowalla
python train.py --trans_loc_file KGE/Graphs/gowalla_scheme2_transe_loc_temporal_100.pkl --trans_interact_file KGE/Graphs/gowalla_scheme2_transe_user-loc_100.pkl --log_file results/log_gowalla --gpu 0
# foursquare
python train.py --dataset checkins-4sq.txt --trans_loc_file KGE/Graphs/foursquare_scheme2_transe_loc_temporal_20.pkl --trans_interact_file KGE/Graphs/foursquare_scheme2_transe_user-loc_100.pkl --log_file results/log_foursquare --gpu 1