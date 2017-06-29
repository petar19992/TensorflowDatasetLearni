#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Model training for Iris data set using Validation Monitor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
OBDII_TRAINING = os.path.join(os.path.dirname(__file__), "dataset.csv")
OBDII_TEST = os.path.join(os.path.dirname(__file__), "datatest.csv")

def main(unused_argv):
  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=OBDII_TRAINING, target_dtype=np.int, features_dtype=np.int)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=OBDII_TEST, target_dtype=np.int, features_dtype=np.int)

  validation_metrics = {
      "accuracy":
          tf.contrib.learn.MetricSpec(
              metric_fn=tf.contrib.metrics.streaming_accuracy,
              prediction_key="classes"),
      "precision":
          tf.contrib.learn.MetricSpec(
              metric_fn=tf.contrib.metrics.streaming_precision,
              prediction_key="classes"),
      "recall":
          tf.contrib.learn.MetricSpec(
              metric_fn=tf.contrib.metrics.streaming_recall,
              prediction_key="classes")
  }
  validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
      test_set.data,
      test_set.target,
      every_n_steps=50,
      #metrics=validation_metrics,
      early_stopping_metric="loss",
      early_stopping_metric_minimize=True,
      early_stopping_rounds=200)

  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.contrib.learn.DNNClassifier(
      feature_columns=feature_columns,
      hidden_units=[10, 20, 10],
      n_classes=3,
      model_dir="/test/test_model",
      config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
#   classifier = tf.contrib.learn.DNNClassifier(
#       feature_columns=feature_columns,
#       hidden_units=[10, 20, 10], 
#       n_classes=3,
#       model_dir="/tmp/test_model")
  # Fit model.
  classifier.fit(x=training_set.data,
                 y=training_set.target,
                 steps=2000,
                 monitors=[validation_monitor])

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(
      x=test_set.data, y=test_set.target)["accuracy"]
  print("Accuracy: {0:f}".format(accuracy_score))

#   print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
#   print(feature_columns)

  tfrecord_serving_input_fn = tf.contrib.learn.build_parsing_serving_input_fn(tf.contrib.layers.create_feature_spec_for_parsing(feature_columns))  
  classifier.export_savedmodel(export_dir_base="test", serving_input_fn = tfrecord_serving_input_fn,as_text=False)

  # Classify two new flower samples.
  new_samples = np.array(
      [[2500, 200, 3, 50], [900, 200, 4, 60]], dtype=int)
  y = list(classifier.predict(new_samples))
  print("Predictions: {}".format(str(y)))


if __name__ == "__main__":
  tf.app.run()
