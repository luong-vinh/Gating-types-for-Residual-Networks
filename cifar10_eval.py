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

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

import cifar10

from tensorflow.python.ops import variables
from layer import BatchNormLayer

import cifar10_model
import network_config

# I need to work on a better way for configuration management. Sacred, perhaps? 
import cifar10_train 

from layer import MywayFFLayer

# TODO: replace these. Check sacred (Alex's suggestion)
#FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string('eval_dir', 'cifar10_eval',
#                            """Directory where to write event logs.""")
# tf.app.flags.DEFINE_string('eval_data', 'test',
#                            """Either 'test' or 'train_eval'.""")
# tf.app.flags.DEFINE_string('checkpoint_dir', 'cifar10_train',
#                            """Directory where to read model checkpoints.""")
# tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
#                             """How often to run the eval.""")
# tf.app.flags.DEFINE_integer('num_examples', 10000,
#                             """Number of examples to run.""")
# tf.app.flags.DEFINE_boolean('run_once', False,
#                          """Whether to run eval only once.""")

# Vinh: setConfig is not an elegant solution. The FLAGS approach
# also has it shortcomings. Will find better ways
# UPDATED: for this program, tf.app.flags is probably good enough
def setConfig():
  config = network_config.getConfig()
  config['eval_dir'] = 'cifar10_eval'
  config['eval_data'] = 'test'
  config['checkpoint_dir'] = 'cifar10_train'
  config['eval_interval_secs'] = 60 * 5
  config['num_examples'] = 10000
  config['run_once'] = False
  config['batch_size'] = 100 # To make sure all the test instances are evaluated

def eval_once(saver, summary_writer, top_k_op, summary_op, ema):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  
  config = network_config.getConfig()
  checkpoint_dir = config['checkpoint_dir']
  num_examples = config['num_examples']
  batch_size = config['batch_size']
  
  
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path: 
      
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
            
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      
      print('Gonna restore for batch norm here')
      # Vinh: restore the moving averages of batch norm's mean and variance.
      # It seems like the normal Saver object doesn't restore the moving averages of
      # untrainable variables
      # Have to hack by copying some ideas in
      # moving_average()'s source code      
      # TODO: temporarily comment out because I forgot to make 
      # change in BatchNorm layer with regards to get_variable()
      # TODO: test with the 2 options: using population mean and variance
      # and mini-batch mean and variance
#       untrainableVars = list(set(variables.all_variables()) - 
#                              set(variables.trainable_variables()))
#        
#       variables_to_restore = {val.name : val     #{ema.average_name(val) : val
#                               for val in untrainableVars}                           
#              
#       batchNormCol = tf.get_collection(BatchNormLayer.batchNormCollectionID)
#       # a bit clunky
#       restoredVarMap =  {ema.average_name(variables_to_restore[key]) 
#                           : variables_to_restore[key] 
#                           for key in batchNormCol
#                           if key in variables_to_restore}
#       batchNormSaver = tf.train.Saver(restoredVarMap)
#       batchNormSaver.restore(sess, ckpt.model_checkpoint_path)
      
      # Vinh: adjust after a new change in BatchNormLayer
      batchNormVarCol = tf.get_collection(BatchNormLayer.batchNormCollectionID)
      restoredVarMap = {ema.average_name(var) : var 
                        for var in batchNormVarCol}
      batchNormSaver = tf.train.Saver(restoredVarMap)
      batchNormSaver.restore(sess, ckpt.model_checkpoint_path)


      # Vinh: So from this point, batch norm's means and variances are restored to their
      # moving average values       
       
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      
      # Vinh: the original tensorflow way will underestimate the true precision
      num_iter = int(math.ceil(num_examples / batch_size))
      true_count = 0  # Counts the number of correct predictions.
      
      # If num_examples is not divisible by batch_size, then total_sample_count
      # will be greater than num_examples. If we assume, in this case, the 
      # filename_queue will be read until the last file, than the precision 
      # calculated here will be less than the correct precision.
      total_sample_count = num_iter * batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: Original precision @ 1 = %.3f' % (datetime.now(), precision))
      
      totalExamplesEvaluated = step * batch_size
      correctPrecision = true_count / totalExamplesEvaluated
      print('The correct precision %.3f over %d evaluated examples is: ' 
            %(correctPrecision, totalExamplesEvaluated))
      
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
      print('Reach here with the total steps: %d' % (step))
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)
    
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  
  config = network_config.getConfig()
  test_or_train = config['eval_data']
  eval_dir = config['eval_dir']
  run_once = config['run_once']
  eval_interval_secs = config['eval_interval_secs']
  
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels for CIFAR-10.
    eval_data = test_or_train == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    #logits = cifar10.inference(images)
    # 20 layer network
    # logits = cifar10_model.buildResidualStyleNetwork(images, is_train_phase = False)
    # 56 layers
#     logits = cifar10_model.buildResidualStyleNetwork(images, is_train_phase = False, 
#                                                      numStackedBlocks = 9)
    # is_train_phase should be false but I need to change BatchNormLayer first
    # so use poor man's batch norm for now
#     logits = cifar10_model.buildNetworkWithVariableScope(images, 
#                                  is_train_phase = True, 
#                                  gateType = MywayFFLayer.HIGHWAY_GATE, 
#                                  numStackedBlocks = 3)
    #logits = cifar10_model.inference(images)    
    
    logits = cifar10_model.buildNetworkWithVariableScope(images, 
                              is_train_phase = False,
                              gateType = MywayFFLayer.RESIDUAL_GATE,
                              numStackedBlocks = 3)
    
#     logits = cifar10_model.buildNetworkWithVariableScope(images, 
#                               is_train_phase = False,
#                               gateType = MywayFFLayer.HIGHWAY_GATE, 
#                               numStackedBlocks = 9)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    # Vinh: disable this for now
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    # Vinh: Will restore weights after I do a fair comparison with ResNet paper
#     variables_to_restore = variable_averages.variables_to_restore()
#     saver = tf.train.Saver(variables_to_restore)    
    saver = tf.train.Saver()
    
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.train.SummaryWriter(eval_dir,
                                            graph_def=graph_def)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op, variable_averages)
      if run_once:
        break      
      time.sleep(eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument  
  cifar10_train.setConfig()
  setConfig()
  eval_dir = network_config.getConfig()['eval_dir']
  cifar10.maybe_download_and_extract()
  if gfile.Exists(eval_dir):
    gfile.DeleteRecursively(eval_dir)
  gfile.MakeDirs(eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
