# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import threading
from kitchen_bot.alarm_manager import AlarmManager


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


class LabelImageAsyc:

    def __init__(self):
        self.graph = None
        self.allowProcessing = False
        self.started = False
        self.file_name = None
        self.read_lock = threading.Lock()

    def runAsync(self, file_name):
        thread = threading.Thread(target=self.run, args=[file_name])
        thread.start()

    def predict(self, file_name):
        with self.read_lock:
            self.file_name = file_name
            self.allowProcessing = True

    def isProcessingAllowed(self):
        with self.read_lock:
            return self.allowProcessing

    def startRunning(self):
        print("predicting")
        while self.started:
            with self.read_lock:
                if self.allowProcessing:
                    continue
                if self.file_name is None:
                    continue
            self.run(self.file_name)

    def run(self, file_name):

        print("running file")

        with self.read_lock:
            self.allowProcessing = False

        model_file = "/Users/oofong/Projects/Hackathon/HackathonProject/kitchen_bot/models/sink_graph.pb"
        label_file = "/Users/oofong/Projects/Hackathon/HackathonProject/kitchen_bot/models/sink_labels.txt"
        input_height = 299
        input_width = 299
        input_mean = 0
        input_std = 255
        input_layer = "Placeholder"
        output_layer = "final_result"

        if self.graph is None:
            print("initializing graph")

            parser = argparse.ArgumentParser()
            parser.add_argument("--image", help="image to be processed")
            parser.add_argument("--graph", help="graph/model to be executed")
            parser.add_argument("--labels", help="name of file containing labels")
            parser.add_argument("--input_height", type=int, help="input height")
            parser.add_argument("--input_width", type=int, help="input width")
            parser.add_argument("--input_mean", type=int, help="input mean")
            parser.add_argument("--input_std", type=int, help="input std")
            parser.add_argument("--input_layer", help="name of input layer")
            parser.add_argument("--output_layer", help="name of output layer")
            args = parser.parse_args()

            if args.graph:
                model_file = args.graph
            if args.image:
                file_name = args.image
            if args.labels:
                label_file = args.labels
            if args.input_height:
                input_height = args.input_height
            if args.input_width:
                input_width = args.input_width
            if args.input_mean:
                input_mean = args.input_mean
            if args.input_std:
                input_std = args.input_std
            if args.input_layer:
                input_layer = args.input_layer
            if args.output_layer:
                output_layer = args.output_layer

            self.graph = load_graph(model_file)
            t = read_tensor_from_image_file(
                file_name,
                input_height=input_height,
                input_width=input_width,
                input_mean=input_mean,
                input_std=input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)

        with tf.Session(graph=self.graph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            })
        results = np.squeeze(results)

        top_name, top_score = None, None

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)

        for i in top_k:
            label, score = labels[i], results[i]

            if top_name is None:
                top_name = label

            if top_score is None:
                top_score = score

            print(label, score)

        AlarmManager().processAlarm(top_name, top_score)
