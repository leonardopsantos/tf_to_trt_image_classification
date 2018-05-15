# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

import sys
sys.path.append('third_party/models/')
sys.path.append('third_party/models/research')
sys.path.append('third_party/models/research/slim')
#from PIL import Image
#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from model_meta import NETS, FROZEN_GRAPHS_DIR, CHECKPOINT_DIR
import time
import cv2
import datetime
import os
import argparse
import multiprocessing

import jetsonTX2Power as tx2pwr

TEST_OUTPUT_PATH='data/test_output_tf.txt'
NUM_RUNS=3

def test(image_list_file, pwr_devices, net_name, net_meta):

    print("Testing %s" % net_name)

    with open(net_meta['frozen_graph_filename'], 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True

    tf_sess = tf.Session(config=tf_config, graph=graph)
    tf_input = tf_sess.graph.get_tensor_by_name(net_meta['input_name'] + ':0')
    tf_output = tf_sess.graph.get_tensor_by_name(net_meta['output_names'][0] + ':0')

    csv_file=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-test_tf_power-")+net_name+'.csv'

    with open(image_list_file, 'r') as f:
        image_list = f.readlines()
    image_list = [x.strip() for x in image_list] 
    path_prefix, path_basename = os.path.split(image_list_file)

    for f_name in image_list:
        image_file_name = path_prefix+'/'+f_name
        if os.path.isfile(image_file_name) != True:
            print "Could not find file: "+image_file_name
            continue

        # load and preprocess image
        image = cv2.imread(image_file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (net_meta['input_width'], net_meta['input_height']))
        image = net_meta['preprocess_fn'](image)

        # run network
        times = []
        for i in range(NUM_RUNS + 1):
            t0 = time.time()
            output = tf_sess.run([tf_output], feed_dict={
            tf_input: image[None, ...]
            })[0]
            t1 = time.time()
            times.append(1000 * (t1 - t0))

        avg_time = np.mean(times[1:]) # don't include first run

        tx2pwr.to_csv(csv_file, pwr_devices, net_name+'-avgTime: '+str(avg_time))

        # parse output
        #top5 = net_meta['postprocess_fn'](output)
        #print(top5)
        with open(TEST_OUTPUT_PATH, 'a+') as test_f:
            test_f.write("%s %s\n" % (net_name, avg_time))

    tx2pwr.to_csv(csv_file, pwr_devices, net_name+':done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measures inference time and power in the Jetson TX2 using TensorRT')
    parser.add_argument('image_list_file', help='Text file with the list of images to be used')
    lineargs = parser.parse_args()

    if os.path.isfile(lineargs.image_list_file) !=  True:
        print "Could not find file"+lineargs.image_list_file
        sys.exit()

    pwr_devices=tx2pwr.create_devices()

    for net_name, net_meta in NETS.items():
        if 'exclude' in net_meta.keys() and net_meta['exclude'] is True:
            continue
        ## Using a subprocess so the CUDA memory gets released! Yes, THIS IS NECESSARY,
        ## as otherwise the python get killed by an allocation error
        p = multiprocessing.Process(target=test, args=(lineargs.image_list_file, pwr_devices, net_name, net_meta,))
        p.start()
        p.join()
