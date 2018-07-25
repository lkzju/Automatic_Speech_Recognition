# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : timit_preprocess.py
# Description  : Feature preprocessing for TIMIT dataset
# ******************************************************

"""
Do MFCC over all *.wav files and parse label file Use os.walk to iterate all files in a root directory

original phonemes:

phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

mapped phonemes(For more details, you can read the main page of this repo):

phn = ['sil', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'el', 'en', 'epi', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'q', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']
"""

import os
import argparse
import glob
import sys
#import sklearn
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
from speechvalley.feature.core import calcfeat_delta_delta, spectrogramPower

## original phonemes
phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

## cleaned phonemes
#phn = ['sil', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'el', 'en', 'epi', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'ix', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'q', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']

def wav2feature(rootdir, save_directory, mode, feature_len, level, keywords, win_len, win_step,  seq2seq, save):
    timit_dir = os.path.abspath(os.path.join(rootdir, os.pardir))
    feat_dir = os.path.join(save_directory, level, keywords, mode)
    label_dir = os.path.join(save_directory, level, keywords, 'label')
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)
    count = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            fullFilename = os.path.join(subdir, file)
            filenameNoSuffix = os.path.splitext(fullFilename)[0]
            if file.endswith('.WAV'):
                file_phn_dir = os.path.split(fullFilename)[0]
                file_wav = os.path.join(timit_dir, "wav")

                file_wav_2 = fullFilename[len(timit_dir) + 1:]
                file_wav = os.path.join(file_wav, file_wav_2)
                file_wav = file_wav.replace(".WAV", ".wav")
                if not os.path.exists(file_wav):
                    continue

                rate = None
                sig = None
                try:
                    (rate,sig)= wav.read(file_wav)
                except ValueError as e:
                    if e.message == "File format 'NIST'... not understood.":
                        print('You should use nist2wav.sh to convert NIST format files to WAV files first, nist2wav.sh is in core folder.')
                        return
                feat = calcfeat_delta_delta(sig,rate,win_length=win_len,win_step=win_step,mode=mode,feature_len=feature_len)
                feat = preprocessing.scale(feat)
                feat = np.transpose(feat)
                print(feat.shape)

                if level == 'phn':
                    labelFilename = os.path.join(file_phn_dir, file.replace(".WAV", ".PHN"))
                    phenome = []
                    with open(labelFilename,'r') as f:
                        if seq2seq is True:
                            phenome.append(len(phn)) # <start token>
                        for line in f.read().splitlines():
                            s=line.split(' ')[2]
                            p_index = phn.index(s)
                            phenome.append(p_index)
                        if seq2seq is True:
                            phenome.append(len(phn)+1) # <end token>
                        print(phenome)
                    phenome = np.array(phenome)

                elif level == 'cha':
                    labelFilename = os.path.join(file_phn_dir, file.replace(".WAV", ".WRD"))
                    phenome = []
                    sentence = ''
                    with open(labelFilename,'r') as f:
                        for line in f.read().splitlines():
                            s=line.split(' ')[2]
                            sentence += s+' '
                            if seq2seq is True:
                                phenome.append(28)
                            for c in s:
                                if c=="'":
                                    phenome.append(27)
                                else:
                                    phenome.append(ord(c)-96)
                            phenome.append(0)

                        phenome = phenome[:-1]
                        if seq2seq is True:
                            phenome.append(29)
                    print(phenome)
                    print(sentence)

                count+=1
                print('file index:',count)
                if save:
                    featureFilename = os.path.join(feat_dir, filenameNoSuffix.split(os.sep)[-2]) + filenameNoSuffix.split(os.sep)[-1]+'.npy'
                    np.save(featureFilename,feat)
                    labelFilename = os.path.join(label_dir, filenameNoSuffix.split(os.sep)[-2]) +filenameNoSuffix.split(os.sep)[-1]+'.npy'
                    print(labelFilename)
                    np.save(labelFilename,phenome)


if __name__ == '__main__':
    # character or phoneme
    #corpus_dir = "C:\\Research\\Corpus\\TIMIT\\"
    #save_dir = "C:\\Research\\Corpus\\InputFiles"
    parser = argparse.ArgumentParser(prog='timit_preprocess',
                                     description="""
                                     Script to preprocess timit data
                                     """)
    parser.add_argument("path", help="Directory where Timit dataset is contained", type=str)
    parser.add_argument("save", help="Directory where preprocessed arrays are to be saved",
                        type=str)
    parser.add_argument("-n", "--name", help="Name of the dataset",
                        choices=['TRAIN', 'DEVELOP', 'CORE_TEST', 'TEST'],
                        type=str, default='TRAIN')
    parser.add_argument("-l", "--level", help="Level",
                        choices=['cha', 'phn'],
                        type=str, default='phn')
    parser.add_argument("-m", "--mode", help="Mode",
                        choices=['mfcc', 'fbank'],
                        type=str, default='mfcc')
    parser.add_argument('--featlen', type=int, default=13, help='Features length')
    parser.add_argument("--seq2seq", help="set this flag to use seq2seq", action="store_false")

    parser.add_argument("-winlen", "--winlen", type=float,
                        default=0.02, help="specify the window length of feature")

    parser.add_argument("-winstep", "--winstep", type=float,
                        default=0.01, help="specify the window step length of feature")

    args = parser.parse_args()
    root_directory = args.path
    save_directory = args.save
    level = args.level
    mode = args.mode
    feature_len = args.featlen
    name = args.name
    seq2seq = args.seq2seq
    win_len = args.winlen
    win_step = args.winstep

    root_directory = os.path.join(root_directory, name)
    if root_directory == ".":
        root_directory = os.getcwd()
    if save_directory == ".":
        save_directory = os.getcwd()
    if not os.path.isdir(root_directory):
        raise ValueError("Root directory does not exist: " + root_directory)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    wav2feature(root_directory, save_directory, mode=mode, feature_len=feature_len,
                level=level, keywords=name, win_len=win_len, win_step=win_step,
                seq2seq=seq2seq, save=True)
