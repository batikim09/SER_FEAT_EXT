# SER_FEAT_EXT
This repository includes source codes and documents explaining feature extraction of speech emotion recognition (https://github.com/batikim09/LIVE_SER).

Maintainer: [**batikim09**](https://github.com/**github-user**/) (**batikim09**) - **j.kim@utwente.nl**

<a id="top"/>

This folder has source codes of feature extraction for speech emotion recognition. Firstly, feature vectors are extracted from each utterance wav file. Secondly, feature vectors are collected, and it generatates cross-validation folds and save them in a H5 database.

##Contents
1. <a href="#1--installation-requirements">Installation Requirements</a>

2. <a href="#2--usage">Usage</a>

3. <a href="#3--references">References</a>

## 1. Installation Requirements <a id="1--installation-requirements"/>
This software only runs on OSX or Linux (tested on Ubuntu). It is compatible with python 2.x and 3.x, but the following descrptions assume that python 3.x is installed.

### basic system packages

This software relies on several system packages that must be installed using a software manager.

For Ubuntu, please run the following steps:

`sudo apt-get install python-pip python-dev libhdf5-dev portaudio19-dev'

### python packages
Using pip, install all pre-required modules.
(pip version >= 8.1 is required, see: http://askubuntu.com/questions/712339/how-to-upgrade-pip-to-latest)

sudo pip3 install -r requirements.txt

## 2. Usage <a id="2--usage"/>

We assume that we download the eNTERFACE corpus that is freely available (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.220.2113&rep=rep1&type=pdf). The corpus location is configurable but we assume it is "./wav/enterface".


### Feature extraction
We explain "./scripts/extract_feat_enterface.txt".

Firstly, a meta information file should be prepared. See an example in "./meta/sanity.enterface.txt".

First column values show wave file paths and other colume values show meta information such as class, gender, and etc.

Secondly, for gain normalisation, we need to collect gain stats by running:

python ./src/extract_feat_temporal_LLD_rosa.py -f ./feat/RAW/ -m ./meta/sanity.enterface.txt --gain_stat

Finally, use collected gain information as:

python ./src/extract_feat_temporal_LLD_rosa.py -f ./feat/MSPEC/ -m ./meta/sanity.enterface.txt -min -1.0541904 -max 1.1097699

without specific arguments, it will generate Mel-spec feature files (.csv) in "./feat/MSPEC/".

For raw wave feature,

python ./src/extract_feat_temporal_LLD_rosa.py -f ./feat/RAW/ -m ./meta/sanity.enterface.txt -min -1.0541904 -max 1.1097699 --wav

For log-spec features,

python ./src/extract_feat_temporal_LLD_rosa.py -f ./feat/LSPEC/ -m ./meta/sanity.enterface.txt -min -1.0541904 -max 1.1097699 --log_spec

The above extraction scripts will generate meta.*.out files that are need for the next step, composing H5DB.


### H5DB composition

The above step extracts feature vectors and store them per wave file. To handle variations in duration and cross-validation, and boost up training speed, we generate a H5 database (https://www.h5py.org/).

See "./scripts/compose_h5db.sh".

python ./src/h5db_builder.py -input ./meta/sanity.enterface.txt.raw.out -m_steps 16000 -c_idx 2 -n_cc 43 -c_len 1600 --two_d -mt 1:3:4:5:6:7 -out ./h5db/ENT.RAW.3cls.av

It generates a h5 database "ENT.RAW.3cls.av" in ./h5db/.
The database will be used in trainer (https://github.com/batikim09/SER_KERAS_TF_TRAINER).

## 3. References <a id="3--references"/>

This software is based on the following papers. Please cite one of these papers in your publications if it helps your research:

@inproceedings{kim2017interspeech,
  title={Towards Speech Emotion Recognition ``in the wild'' using Aggregated Corpora and Deep Multi-Task Learning},
  author={\textbf{Kim, Jaebok} and Englebienne, Gwenn and Truong, Khiet P and Evers, Vanessa},
  booktitle={Proceedings of the INTERSPEECH},
  pages={1113--1117},
  year={2017}
}


@inproceedings{kim2017acmmm, title={Deep Temporal Models using Identity Skip-Connections for Speech Emotion Recognition}, author={Kim, Jaebok and Englebienne, Gwenn and Truong, Khiet P and Evers, Vanessa}, booktitle={Proceedings of ACM Multimedia}, pages={1006-1013}, year={2017} }

@inproceedings{kim2017acii, title={Learning spectro-temporal features with 3D CNNs for speech emotion recognition}, author={Kim, Jaebok and Truong, Khiet and Englebienne, Gwenn and Evers, Vanessa}, booktitle={Proceedings of International Conference on Affective Computing and Intelligent Interaction}, pages={}, year={2017} }

<a id="top"/> 
