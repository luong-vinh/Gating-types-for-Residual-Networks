import tensorflow as tf

from layer import ConvLayer, FFLayer, MyWayConvLayer, StackedMyWayConvLayer, \
                  BatchNormedConvLayer, StackedBatchNormedConvLayer, \
                  createStackedConvLayer, MywayFFLayer
                  
from weight_initialization import WeightDecayInitializer
import cifar10_input

import cifar10
import re
import network_config

NUM_CLASSES = cifar10_input.NUM_CLASSES


# Will add some more parameters such as gateType, is_train etc... here
def buildResidualStyleNetwork(images, is_train_phase = True,
                              gateType = MywayFFLayer.RESIDUAL_GATE, 
                              numStackedBlocks = 3): # n in (6n + 2) layers for Residual Network
  
  # TODO: check again the weight initialization in He et. al.
  wdInit = WeightDecayInitializer(stddev = 1e-4, weightDecay = 1e-4)
  
  ema = tf.train.ExponentialMovingAverage(decay = cifar10.MOVING_AVERAGE_DECAY)
  
  #numStackedBlocks = 3 
  
  
  # Don't use variable_scope or name_scope for now because it seems to cause problems when 
  # combining with tf.Variable() and ExponentialMovingAverage.apply(). Apparently,
  # these scope mechanisms are meant to work with tf.get_variable() only
  # TODO: add get_variable as a way to initialize my variables for layers. It's nice
  # to have some namespace to display in Graphs
    
  # The first stack has (2 * numStackedBlocks + 1) layers (3x3), which consists
  # of numStackedBlocks. Each block can be of type ResNet layer, or Highway layer
  # or the more general Myway layer
  #with tf.variable_scope("stack_1"):
    # First conv layer
  conv1 = BatchNormedConvLayer(images, ema, kernel_shape = [3, 3, 3, 16],
                               strides = [1, 1, 1, 1],
                               kernelInit = wdInit.weightConvLayerInit,
                               padding = 'SAME',
                               operator = tf.nn.relu)
    
  blockInput = conv1
  for _ in xrange(numStackedBlocks):      
    blockInput = StackedBatchNormedConvLayer(blockInput, ema,
                              kernel_shape = [3, 3, 16, 16],
                              strides = [1, 1, 1, 1],
                              kernelInit = wdInit.weightConvLayerInit,
                              gateType = gateType,
                              layerType = BatchNormedConvLayer,
                              numStackedLayers = 2,
                              is_train_phase = is_train_phase)

  # Reduce the size of each map to half      
  #with tf.variable_scope("stack_2"):
#     conv2 = BatchNormedConvLayer(blockInput, ema, kernel_shape = [3, 3, 16, 32],
#                                  strides = [1, 2, 2, 1],
#                                  kernelInit = wdInit.weightConvLayerInit,
#                                  padding = 'SAME')
    
  blockInput = createStackedConvLayer(blockInput, ema, 
                              kernel_shape = [3, 3, 32, 32],
                              strides = [1, 2, 2, 1],
                              kernelInit = wdInit.weightConvLayerInit,
                              padding = 'SAME',
                              gateType = gateType)
  
  numBlocks = numStackedBlocks - 1
  for _ in xrange(numBlocks):
    blockInput = StackedBatchNormedConvLayer(blockInput, ema,
                              kernel_shape = [3, 3, 32, 32],
                              strides = [1, 1, 1, 1],
                              kernelInit = wdInit.weightConvLayerInit,
                              gateType = gateType,
                              layerType = BatchNormedConvLayer,
                              is_train_phase = is_train_phase,
                              numStackedLayers = 2)
  
  #with tf.variable_scope("stack_3"):
  blockInput = createStackedConvLayer(blockInput, ema, 
                              kernel_shape = [3, 3, 64, 64],
                              strides = [1, 2, 2, 1],
                              kernelInit = wdInit.weightConvLayerInit,
                              padding = 'SAME',
                              gateType = gateType)
  
  numBlocks = numStackedBlocks - 1
  for _ in xrange(numBlocks):
    blockInput = StackedBatchNormedConvLayer(blockInput, ema,
                            kernel_shape = [3, 3, 64, 64],
                            strides = [1, 1, 1, 1],
                            kernelInit = wdInit.weightConvLayerInit,
                            gateType = gateType,
                            layerType = BatchNormedConvLayer,
                            is_train_phase = is_train_phase,
                            numStackedLayers = 2)
  
  # global average pool
  # I wonder which one is faster: reduce_mean or avg_pool
  # Will check later. 
  inputHeight, inputWidth = blockInput.get_shape().as_list()[1:3]
  # Note that the ksize's order is different from that of kernel_shape in
  # convolution  
  globalPoolOutput = tf.nn.avg_pool(blockInput, [1, inputHeight, inputWidth, 1], 
                                    [1, inputHeight, inputWidth, 1], 
                                    padding = 'SAME')
  
  
  # Collapse 4D tensor to 2D
  numFilters = globalPoolOutput.get_shape().as_list()[3]
  # We know that the shape of the global pool output is
  # [batch_size, 1, 1, numFilters]
  poolOutput = tf.reshape(globalPoolOutput, [-1, numFilters])
  
  # set weight decay. Will check this later
  # TODO: check weight initialization and weight decay here for 
  # this layer and others
  wdInit.setParams(1.0 / numFilters, 0.0)
  with tf.variable_scope("softmax_linear"):
    softmax_linear = FFLayer(poolOutput, NUM_CLASSES, 
                             weightInit = wdInit.weightDecayFFLayerInit,
                             operator = None)
  
  return softmax_linear
    

# To support multi-GPUs. Ideally, I should have modified the function above
# to support both ways of getting a variable: tf.Variable() and tf.get_variable.
# UPDATE: after the change I made in BatchNormConvLayer, it's much easier now 
# to combine these two functions into one.
def buildNetworkWithVariableScope(images, is_train_phase = True,
                              gateType = MywayFFLayer.RESIDUAL_GATE, 
                              numStackedBlocks = 3): # n in (6n + 2) layers for Residual Network
  print("Going to build myway network now")
  # TODO: check again the weight initialization in He et. al.
  wdInitializer = WeightDecayInitializer(stddev = 1e-4, 
                                  weightDecay = 1e-4,
                                  useGetVar = True,
                                  device = '/cpu:0')
  
  # Use He et al's weight initialization
  kernelInitFunc = wdInitializer.weightConvLayerInit2
  
  # TODO check momentum 0.9
  #ema = tf.train.ExponentialMovingAverage(decay = cifar10.MOVING_AVERAGE_DECAY)
  ema = None # for compatibility with the old interface, which I haven't got time to change  
  
  # Don't use variable_scope or name_scope for now because it seems to cause problems when 
  # combining with tf.Variable() and ExponentialMovingAverage.apply(). Apparently,
  # these scope mechanisms are meant to work with tf.get_variable() only  
    
  # The first stack has (2 * numStackedBlocks + 1) layers (3x3), which consists
  # of numStackedBlocks. Each block can be of type ResNet layer, or Highway layer
  # or the more general Myway layer
  
  # With my current method, I don't need to use variable scope here. But I 
  # use it for ease of examining the graph in TensorBoard
  
  with tf.variable_scope('conv1'):
    conv1 = BatchNormedConvLayer(images, ema, kernel_shape = [3, 3, 3, 16],
                                 strides = [1, 1, 1, 1],
                                 kernelInit = kernelInitFunc,
                                 padding = 'SAME',
                                 operator = tf.nn.relu,
                                 use_scope = True,
                                 scope_name = 'first_layer')
  
  blockInput = conv1
  with tf.variable_scope("stack_1"):    
    for blockIndx in xrange(numStackedBlocks):   
      scope_name = 'block_' + str(blockIndx)           
      blockInput = StackedBatchNormedConvLayer(blockInput, ema,
                                kernel_shape = [3, 3, 16, 16],
                                strides = [1, 1, 1, 1],
                                kernelInit = kernelInitFunc,
                                gateType = gateType,
                                layerType = BatchNormedConvLayer,
                                numStackedLayers = 2,
                                is_train_phase = is_train_phase,
                                use_scope = True,
                                scope_name = scope_name)

  
  with tf.variable_scope("stack_2"):
    with tf.variable_scope("transition_block"):  
      blockInput = createStackedConvLayer(blockInput, ema, 
                                  kernel_shape = [3, 3, 32, 32],
                                  strides = [1, 2, 2, 1],
                                  kernelInit = kernelInitFunc,
                                  padding = 'SAME',
                                  gateType = gateType,
                                  use_scope= True,
                                  scope_name = 'trans_blk')
    with tf.variable_scope('rest_of_stack'):
      numBlocks = numStackedBlocks - 1
      for blockIndx in xrange(numBlocks):
        scope_name = 'block_' + str(blockIndx)
        blockInput = StackedBatchNormedConvLayer(blockInput, ema,
                                  kernel_shape = [3, 3, 32, 32],
                                  strides = [1, 1, 1, 1],
                                  kernelInit = kernelInitFunc,
                                  gateType = gateType,
                                  layerType = BatchNormedConvLayer,
                                  is_train_phase = is_train_phase,
                                  numStackedLayers = 2,
                                  use_scope= True,
                                  scope_name = scope_name)
  
  with tf.variable_scope("stack_3"):
    with tf.variable_scope("transition_block"):
      blockInput = createStackedConvLayer(blockInput, ema, 
                                  kernel_shape = [3, 3, 64, 64],
                                  strides = [1, 2, 2, 1],
                                  kernelInit = kernelInitFunc,
                                  padding = 'SAME',
                                  gateType = gateType,
                                  use_scope= True,
                                  scope_name = 'trans_blk')
    
    with tf.variable_scope('rest_of_stack'):  
      numBlocks = numStackedBlocks - 1
      for blockIndx in xrange(numBlocks):
        scope_name = 'block_' + str(blockIndx)
        blockInput = StackedBatchNormedConvLayer(blockInput, ema,
                                kernel_shape = [3, 3, 64, 64],
                                strides = [1, 1, 1, 1],
                                kernelInit = kernelInitFunc,
                                gateType = gateType,
                                layerType = BatchNormedConvLayer,
                                is_train_phase = is_train_phase,
                                numStackedLayers = 2,
                                use_scope= True,
                                scope_name = scope_name)
  
  # global average pool
  # I wonder which one is faster: reduce_mean or avg_pool
  # Will check later. 
  inputHeight, inputWidth = blockInput.get_shape().as_list()[1:3]
  # Note that the ksize's order is different from that of kernel_shape in
  # convolution  
  globalPoolOutput = tf.nn.avg_pool(blockInput, [1, inputHeight, inputWidth, 1], 
                                    [1, inputHeight, inputWidth, 1], 
                                    padding = 'SAME')  
  
  # Collapse 4D tensor to 2D
  numFilters = globalPoolOutput.get_shape().as_list()[3]
  # We know that the shape of the global pool output is
  # [batch_size, 1, 1, numFilters]
  poolOutput = tf.reshape(globalPoolOutput, [-1, numFilters])  
  
  with tf.variable_scope("softmax_linear"):
    softmax_linear = FFLayer(poolOutput, NUM_CLASSES, 
                             weightInit = wdInitializer.weightDecayFFLayerInit2,
                             operator = None)
  
  return softmax_linear
  
      