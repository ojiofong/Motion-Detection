# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile
import threading
import numpy as np
from six.moves import urllib
import tensorflow as tf
import cv2
import time

from kitchenbot.alarm_manager import AlarmManager
import SlidingWindow as sw

FLAGS = None

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

IS_GRAPH_DEF_ADDED = False

FLAGS_model_dir = '/Users/oofong/Projects/Hackathon/HackathonProject/kitchenbot/model'
FLAGS_num_top_predictions = 5

alarm = AlarmManager()


# region NodeLookup

class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                FLAGS_model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                FLAGS_model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


# endregion

class ClassifyImage:

    def __init__(self):
        print("init ClassifyImage")
        self.x = 0

    def create_graph(self):
        """
        Creates a graph from saved GraphDef file and returns a saver.
        """

        # Creates graph from saved graph_def.pb.
        with tf.gfile.GFile(os.path.join(FLAGS_model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def run_inference_on_image_async(self, imagePath):
        thread = threading.Thread(target=self.run_inference_on_image, args=([imagePath]))
        thread.start()

    def run_inference_on_image(self, imagePath):
        """Runs inference on an image.

        Args:
          imagePath: Image file name.

        Returns:
          Nothing
        """

        # Creates graph from saved GraphDef.
        self.create_graph()

        human_string, score = self.predict(imagePath)

        # check if valid dish washer
        if alarm.isWindowSlideNeeded(human_string, score):
            print("running window slide to look for plates, cups etc")

            start = time.time()

            image = cv2.imread(imagePath)

            # Optionally scale down the image pixel size for faster sliding
            scale = 0.2
            image = cv2.resize(image, None, fx=scale, fy=scale)
            print("image scaled down for sliding", image.shape)

            # Define the window width/height relative to the image width's size
            relativeSize = int(image.shape[0] / 3)
            (winW, winH) = (relativeSize, relativeSize)
            # (winW, winH) = (128, 128)

            # loop over the sliding window for each layer of the pyramid

            for (x, y, window) in sw.sliding_window(image, stepSize=32, windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue

                # Process window frame if needed here for e.g. using a ML classifier
                # Lousy way to process this. Shouldn't have to write to disk
                cv2.imwrite(imagePath, window)
                human_string, score = self.predict(imagePath)
                alarm.processAlarm(human_string, score)
            print("End sliding window", time.time() - start)
        else:
            alarm.processAlarm(human_string, score)

    def predict(self, imagePath):

        if not tf.gfile.Exists(imagePath):
            tf.logging.fatal('File does not exist %s', imagePath)
        image_data = tf.gfile.GFile(imagePath, 'rb').read()  # jpg file

        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            # Creates node ID --> English string lookup.
            node_lookup = NodeLookup()

            human_string = ''
            score = 0

            top_k = predictions.argsort()[-FLAGS_num_top_predictions:][::-1]
            for node_id in top_k:
                human_string = node_lookup.id_to_string(node_id)
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))
                break  # grab the first data

            return human_string, score

    def maybe_download_and_extract(self):
        """Download and extract model tar file."""
        dest_directory = FLAGS_model_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                    filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def test(self):
        # skip download. Already downloaded in model/ directory
        # maybe_download_and_extract()

        imagePath = '/Users/oofong/Projects/Hackathon/HackathonProject/kitchenbot/images/sink11.jpg'
        self.run_inference_on_image(imagePath)

# ClassifyImage().test()
