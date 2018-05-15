# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

import sys
#sys.path.append("../scripts")
#sys.path.append(".")
from model_meta import NETS
import os
import subprocess
import pdb
import argparse
import os.path
import sys

TEST_OUTPUT_PATH='data/test_output_trt.txt'
TEST_EXE_PATH='./build/src/test/tensorrt'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Measures inference time and power in the Jetson TX2 using TensorRT')
    parser.add_argument('image_list_file', help='Text file with the list of images to be used')
    lineargs = parser.parse_args()

    if os.path.isfile(lineargs.image_list_file) !=  True:
        print "Could not find file"+lineargs.image_list_file
        sys.exit()

    # delete output file 
    if os.path.isfile(TEST_OUTPUT_PATH):
       os.remove(TEST_OUTPUT_PATH)

    for net_name, net_meta in NETS.items():
        if 'exclude' in net_meta.keys() and net_meta['exclude'] is True:
            continue

        if os.path.isfile(net_meta['plan_filename']) !=  True:
            print "Could not find file"+net_meta['plan_filename']
            continue

        args = [
            lineargs.image_list_file,
            net_meta['plan_filename'],
            net_meta['input_name'],
            str(net_meta['input_height']),
            str(net_meta['input_width']),
            net_meta['output_names'][0],
            str(net_meta['num_classes']), 
            net_meta['preprocess_fn'].__name__,
            str(3), # numRuns
            "half", # dataType 
            str(1), # maxBatchSize 
            str(1 << 20), # workspaceSize 
            str(0), # useMappedMemory 
            TEST_OUTPUT_PATH,
            net_name
        ]
        print("Running %s" % net_name)
        subprocess.call([TEST_EXE_PATH] + args)
