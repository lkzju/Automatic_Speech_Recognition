# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : nist2wav.py
# Description  : Converting nist format to wav format for Automatic Speech Recognition
# ******************************************************


from __future__ import print_function
#import subprocess
import os


def nist2wav(src_dir):
    print(" trying to run nist2wav(): " + src_dir)
    count = 0
    for subdir, dirs, files in os.walk(src_dir):
        print("  searching the dir: " + subdir)
        for f in files:
            fullFilename_input = os.path.join(subdir, f)
            if f.endswith('.WAV'):
                count += 1
                mid_dir = subdir[len(src_dir)+1:]
                fullFilename_output = os.path.join(src_dir, "wav")
                fullFilename_output = os.path.join(fullFilename_output, mid_dir)
                if not os.path.exists(fullFilename_output):
                    os.makedirs(fullFilename_output)

                fullFilename_output = os.path.join(fullFilename_output, f[0:f.find(".")]+".wav")

                sph2pipe_path = os.path.split(os.path.realpath(__file__))[0]
                sph2pipe_path = os.path.join(sph2pipe_path, "sph2pipe")
                os.system(sph2pipe_path + " -f rif " + fullFilename_input+" " +fullFilename_output)
                print(fullFilename_output)

if __name__ == '__main__':
    #nist2wav('/home/pony/wsj/')
    #nist2wav("C:\\Research\\Corpus\\TIMIT")
    nist2wav("/home/kli/Corpus/TIMIT")
