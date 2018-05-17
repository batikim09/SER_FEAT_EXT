
#collect gain stats.
python ./src/extract_feat_temporal_LLD_rosa.py -f ./feat/RAW/ -m ./meta/sanity.enterface.txt --gain_stat

#use collected gain information
#default feature extract: Mel-spec
python ./src/extract_feat_temporal_LLD_rosa.py -f ./feat/MSPEC/ -m ./meta/sanity.enterface.txt -min -1.0541904 -max 1.1097699

#feature extract: RAW
python ./src/extract_feat_temporal_LLD_rosa.py -f ./feat/RAW/ -m ./meta/sanity.enterface.txt -min -1.0541904 -max 1.1097699 --wav

#feature extract: Log-spec, deprecation should be fixed.
python ./src/extract_feat_temporal_LLD_rosa.py -f ./feat/LSPEC/ -m ./meta/sanity.enterface.txt -min -1.0541904 -max 1.1097699 --log_spec

#The above extraction scripts will generate a meta.*.out file that is need for composing H5DB.
