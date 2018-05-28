#Compose each feature DB by specifying arguments.
#Speaker-independent cross-validations 
python ./src/h5db_builder.py -input ./meta/sanity.enterface.txt.raw.out -m_steps 16000 -c_idx 2 -n_cc 43 -c_len 1600 --two_d -mt 1:3:4:5:6:7 -out ./h5db/ENT.RAW.3cls.av
python ./src/h5db_builder.py -input ./meta/sanity.enterface.txt.mspec.out -m_steps 100 -c_idx 2 -n_cc 43 -c_len 10 --two_d -mt 1:2:4:5:6:7 -out ./h5db/ENT.MSPEC.2d.3cls.av

#3D feature structure for 3DCNN models
python ./src/h5db_builder.py -input ./meta/sanity.enterface.txt.mspec.out -m_steps 100 -c_idx 2 -n_cc 43 -c_len 10 --three_d -mt 1:2:4:5:6:7 -out ./h5db/ENT.MSPEC.3d.3cls.av

#The following script composes a feature DB by using aggregated corpora.
#specify corpus ID (c_idx) and choose multiple corpora indices  (c_ids)
#To keep speaker independence, specify the index of speaker column (s_idx).
python ./src/h5db_builder_cc_sid.py -input ./meta/sanity.aibo_enterface.txt.mspec.out -m_steps 100 -c_ids 0,1 -c_idx 3 -s_idx 2 -c_len 10 --two_d -mt 1:2:4:5:6:7 -out ./h5db/AE.RAW.2d.3cls.av

