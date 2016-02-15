# Copyright 2015 Google Inc. All Rights Reserved.
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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#from tensorflow.models.image.cifar10 import cifar10
import cifar10
import network_config
import cifar10_model

from layer import MywayFFLayer
# 
# FLAGS = tf.app.flags.FLAGS
# 
# tf.app.flags.DEFINE_string('train_dir', 'cifar10_data',
#                            """Directory where to write event logs """
#                            """and checkpoint.""")
# tf.app.flags.DEFINE_integer('max_steps', 1000000,
#                             """Number of batches to run.""")
# tf.app.flags.DEFINE_boolean('log_device_placement', False,
#                             """Whether to log device placement.""")

# Replace the inflexible tf.app.flags

# Vinh: TODO: same as cifar10_train. I just haven't got time to create some kinds of 
# option processing to get rid of this redundancy

def setConfig():    
    config = network_config.getConfig()
    config['train_dir'] = 'cifar10_train_highway'
    config['max_steps'] = 78125 # 200 epochs # 64000 #1000000 # Vinh: use ResNet's max steps
    config['log_device_placement'] = False
    config['batch_size'] = 128
    config['data_dir'] = 'cifar10_data'


def train():
  #setConfig() # Already set in main
  config = network_config.getConfig()
  train_dir = config['train_dir']
  max_steps = config['max_steps']
  log_device_placement = config['log_device_placement']
  batch_size = config['batch_size']
    
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.       
    
    # 20 layer network
#     logits = cifar10_model.buildResidualStyleNetwork(images, is_train_phase = True)
                                                     
    # 56 layers
#     logits = cifar10_model.buildResidualStyleNetwork(images, is_train_phase = True, 
#                                                      numStackedBlocks = 9)

    
    logits = cifar10_model.buildNetworkWithVariableScope(images, 
                          is_train_phase = True, 
                          gateType = MywayFFLayer.HIGHWAY_GATE, 
                          numStackedBlocks = 9)
    
    # Calculate loss.
    loss = cifar10.loss(logits, labels)
    print('Loss used logits from cifar10_model')
    
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.    
    lr = tf.placeholder(tf.float32, [], "learning_rate")
    train_op = cifar10.train(loss, global_step, lr)   

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=log_device_placement))
    sess.run(init)

    # Create a saver.
    # Vinh: this should save the moving averages of batch mean and variance
    # =============== Maybe not, I need to check it now ============================= 
    saver = tf.train.Saver(tf.all_variables())
    #saver = tf.train.Saver() # What is the difference here?


    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(train_dir,
                                            graph_def=sess.graph_def)    
    
    for step in xrange(max_steps):
      start_time = time.time()
      
      # Vinh: change learning rates at steps 32k and 48k, terminating
      # at step 64k (counts from 1) (as in ResNet paper)      
      feed_dict = {lr : 0.1}      
      # Not reducing the learning rate for now
      if (step + 1) == 32000:
        feed_dict = {lr : 0.01}
      elif (step + 1) == 48000:
        feed_dict = {lr : 0.001}
      _, loss_value = sess.run([train_op, loss], feed_dict = feed_dict)
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        # Vinh: If I monitor the learning rate in cifar10.py, I'd need to
        # pass the feed_dict above to the running of summary_op 
        summary_str = sess.run(summary_op) 
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == max_steps:
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  # Have to set config first
  # TODO: remove the need for this, will check how Python initialize a module
  setConfig()
  cifar10.maybe_download_and_extract()
  config = network_config.getConfig()
  train_dir = config['train_dir']
  if gfile.Exists(train_dir):
    gfile.DeleteRecursively(train_dir)
  gfile.MakeDirs(train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
