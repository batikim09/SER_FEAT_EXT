import os.path
import os
import sys
import argparse
import numpy as np
import librosa

# To change this template, choose Tools | Templates
from os.path import join, getsize
from feat_ext import *		

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--feat_folder", dest= 'feat_folder', type=str, help="feat folder", default='./feat/')
parser.add_argument("-m", "--meta_file", dest= 'meta_file', type=str, help="meta file", default='./meta.txt')
parser.add_argument("-min", "--min", dest= 'min', type=float, help="min")
parser.add_argument("-max", "--max", dest= 'max', type=float, help="max")

parser.add_argument("--pca_log_spec", help="PCA whitened log spectrogram", action="store_true")
parser.add_argument("--log_spec", help="log spectrogram", action="store_true")
parser.add_argument("--wav", help="raw wave", action="store_true")
parser.add_argument("--gain_stat", help="collect gain stats", action="store_true")


args = parser.parse_args()

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)
  
feat_folder_prefix = args.feat_folder

if not os.path.exists(feat_folder_prefix):
    os.makedirs(feat_folder_prefix)
    
if args.min and args.max:
	min_max = (args.min, args.max)
else:
	min_max = None

feat_ext = FeatExt(min_max)

#initial gain thresholds
min_gain = np.finfo(np.float32).max
max_gain = np.finfo(np.float32).min

if args.wav:
	new_meta_file_name = args.meta_file + ".raw.out"
elif args.log_spec:
	new_meta_file_name = args.meta_file + ".lspec.out"
elif args.pca_log_spec:
	new_meta_file_name = args.meta_file + ".pspec.out"
else:
	new_meta_file_name = args.meta_file + ".mspec.out"

new_meta_file = open(new_meta_file_name, "w")

#load meta file
meta_map = {}
if args.meta_file != None:
	
	meta_file = open(args.meta_file)

	count = 0
	for line in meta_file:
		if count == 0:
			count = count + 1
			new_meta_file.write(line.rstrip() + "\tfeatfile\n")
			continue

		line = line.rstrip('\n')
		meta_info = line.split('\t')

		file_args = meta_info[0].split('/')
		
		feat_file = feat_folder_prefix + "/"  + file_args[len(file_args) - 1] + ".csv"

		count = count + 1

		if args.gain_stat:
			y, sr = librosa.load(meta_info[0])
			print("sr: ", str(sr))
			min = np.min(y)
			max = np.max(y)

			if min < min_gain:
				min_gain = min
			if max > max_gain:
				max_gain = max	
		else:
			if args.log_spec:
				feat_ext.extract_log_spectrogram_file(meta_info[0], file = feat_file)
			elif args.pca_log_spec:
				feat_ext.extract_pca_logspec_file(meta_info[0], file = feat_file)
			elif args.wav:
				feat_ext.extract_wav_file(meta_info[0], file = feat_file)
			else:
				feat_ext.extract_melspec_file(meta_info[0], file = feat_file, n_mels = 80)
		
		print("# " + str(count) + "file: " +  str(meta_info[0]) + " -> " + feat_file)

		new_meta_line = ""
		for meta in meta_info:
			new_meta_line += meta + '\t'
		new_meta_line += feat_file + '\n'
		
		new_meta_file.write(new_meta_line)
		
	print('total utterances in meta map: ' + str(count))
	new_meta_file.close()

	if args.gain_stat:
		line = 'min gain:\t' + str(min_gain) + ", max gain:\t" + str(max_gain) + '\n'
		print(line)
		stat_file = open(args.meta_file + ".gain.txt", "w")
		stat_file.write(line)
		stat_file.close()
