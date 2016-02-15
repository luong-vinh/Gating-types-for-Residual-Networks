import tensorflow as tf

# Use math for now, will check later if really necessary
from math import sqrt
#import numpy as np



# TODO: sure, there is plenty of room for improvement by refactoring my code here.
def xavierFFLayerInit(ffLayer):
    input_size = ffLayer.getInputSize()
    output_size = ffLayer.getOuttopputSize()
    op = ffLayer.getOperator()
    return xavierInit(input_size, output_size, op)

# Using He et al's reasoning for sigmoid activation is quite involving, which
# I haven't got time to work out. Just use xavier way for now
def xavierConvLayerInit(convLayer):
  kernel_shape = convLayer.getKernelShape()  
  input_size = kernel_shape[0] * kernel_shape[1]
  output_size = 1
  numOutChannels = kernel_shape[3]
  
  op = convLayer.getOperator()
  
  low = - sqrt(6. / (input_size + output_size))
  # If it's sigmoid, multiply the range by 4. For other types of activations,
  # just keep the same range as in Theano
  if op is tf.nn.sigmoid:
      low *= 4   
  high = - low
  
  biasInitVal = convLayer.getBiasInitVal()
  
  # TODO: hard code device and get_variable
  with tf.device('/cpu:0'):
    initializer = tf.random_uniform_initializer(low, high, dtype = tf.float32)
    weights = tf.get_variable('weights', kernel_shape, initializer= initializer)    
    
    biases = tf.get_variable('biases', [numOutChannels], tf.float32,
                           tf.constant_initializer(biasInitVal))    
  
    return weights, biases  

def xavierInit(input_size, output_size, op, biasInitVal = 0.0):
    low = - sqrt(6. / (input_size + output_size))
    # If it's sigmoid, multiply the range by 4. For other types of activations,
    # just keep the same range as in Theano
    if op is tf.nn.sigmoid:
        low *= 4 
    
    high = - low
    weight = tf.Variable(tf.random_uniform([input_size, output_size], low, high))
    # Do I need a separate function for bias init?
    bias = tf.Variable(tf.zeros([output_size]))    
    return weight, bias 




# Use get_variable()
def xavierInit2(layer):
    pass



# Make a weight initializer here
# NOTE: Read tensorflow's howto on sharing variables before using
# get_variable option. 
# Currently, device parameter is only for the case where 
# get_variable() is enabled. I need to find out whether
# having device parameter at this low level is a good design.
# TODO: a lot of code repetition: move the bias initalization
# to a separate function
class WeightDecayInitializer(object):
    def __init__(self, stddev = 1.0, weightDecay = 0.0, 
                 useGetVar = False,
                 device = '/cpu:0'):
        self._stddev = stddev       
        self._weightDecay = weightDecay
        self._useGetVar = useGetVar 
        self._device = device
        
    
    def setParams(self, stddev, weightDecay):        
        self._stddev = stddev
        self._weightDecay = weightDecay    
        
    def useGetVar(self):
      self._useGetVar = True    
    
    def weightDecayFFLayerInit(self, ffLayer):
        input_size = ffLayer.getInputSize()
        output_size = ffLayer.getOutputSize()
        biasInitVal = ffLayer.getBiasInitVal()               
        
        weights = variableWithWeightDecay('weights', [input_size, output_size],
                                          self._stddev, 
                                          self._weightDecay,
                                          self._useGetVar,
                                          self._device)         
        # No decaying for biases
        # Will write a more reusable version for bias and weights later
        if not self._useGetVar:          
          biasVals = tf.constant(biasInitVal, dtype= tf.float32, 
                                 shape = [output_size])               
          biases = tf.Variable(biasVals, name = 'biases')                    
        else:  
          with tf.device(self._device): 
            biases = tf.get_variable('biases', [output_size], 
                                     tf.float32, tf.constant_initializer(biasInitVal))
            # debug
            #biases = _variable_on_cpu('biases', [output_size], tf.constant_initializer(biasInitVal))          
        

        return weights, biases

    # Use cifar10.py for now. Gonna write my own later     
    def weightConvLayerInit(self, convLayer):
        kernel_shape = convLayer.getKernelShape()                   
        kernel = variableWithWeightDecay('weights', kernel_shape, 
                                         self._stddev,
                                         self._weightDecay, 
                                         self._useGetVar,
                                         self._device)
        
        numOutputs = kernel_shape[3] # Tensorflow's convention             
        biasInitVal = convLayer.getBiasInitVal()
        
        if not self._useGetVar:
          biasVals = tf.constant(biasInitVal, dtype= tf.float32, shape = [numOutputs])               
          biases = tf.Variable(biasVals, name = 'biases')
        else: 
          with tf.device(self._device):
            biases = tf.get_variable('biases', [numOutputs], 
                                     tf.float32, 
                                     tf.constant_initializer(biasInitVal))
          
        #biases = _variable_on_cpu('biases', [numOutputs], tf.constant_initializer(biasInitVal))
        
        return kernel, biases
    
    # He et al's weight initalization    
    def weightDecayFFLayerInit2(self, ffLayer):
      weights = heEtAlWeightFFInit(ffLayer, self._weightDecay, 
                                   self._useGetVar, self._device)      
      output_size = ffLayer.getOutputSize()
      biasInitVal = ffLayer.getBiasInitVal() 
      # No decaying for biases
      # Will write a more reusable version for bias and weights later
      if not self._useGetVar:          
        biasVals = tf.constant(biasInitVal, dtype= tf.float32, 
                               shape = [output_size])               
        biases = tf.Variable(biasVals, name = 'biases')
        # For some reasons, without the '/cpu:0' here, tensorflow
        # can't compute gradient. Something wrong
      else:  
        with tf.device(self._device): 
          biases = tf.get_variable('biases', [output_size], 
                                   tf.float32, tf.constant_initializer(biasInitVal))
          
      return weights, biases
    
    # He et al's weight initalization
    def weightConvLayerInit2(self, convLayer):
      kernel = heEtAlWeightConvInit(convLayer, self._weightDecay, 
                                    self._useGetVar, self._device)
      kernel_shape = convLayer.getKernelShape()
      numOutputs = kernel_shape[3] # Tensorflow's convention             
      biasInitVal = convLayer.getBiasInitVal()
      
      if not self._useGetVar:
        biasVals = tf.constant(biasInitVal, dtype= tf.float32, 
                               shape = [numOutputs])               
        biases = tf.Variable(biasVals, name = 'biases')
      else: 
        with tf.device(self._device):
          biases = tf.get_variable('biases', [numOutputs], 
                                   tf.float32, 
                                   tf.constant_initializer(biasInitVal))
      
      return kernel, biases
     
# when I use get_variable(), I can't control whether the variables 
# will be shared or not, except when I have the access to the top level of
# variable scopes. 
# I just write a particular variable with weight decay here   
def variableWithWeightDecay(name, shape, stddev, weightDecay, 
                            useGetVar = False,
                            device = '/cpu:0'):
    if not useGetVar: # Create new variable
        intVal = tf.truncated_normal(shape, stddev = stddev)
        var = tf.Variable(intVal, name = name)    
    else: # depend on the top level variable scope      
      with tf.device(device): 
        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable(name, shape, initializer = initializer)  
    
    if weightDecay:
        weightDecayLoss = tf.mul(tf.nn.l2_loss(var), weightDecay, 
                                 name = 'weight_decay_loss')
        tf.add_to_collection('losses', weightDecayLoss)            
    return var   

# TODO: consider whether I should include the parameter device here 
# TODO: separate weight initialization and weight decay  
# Weight initialization follows He et al. (Delving Deep into Rectifiers: 
# Surpassing Human-Level Performance on ImageNet Classification (2015))
def heEtAlWeightConvInit(convLayer, 
                         weightDecay,
                         useGetVar = True,
                         device = '/cpu:0'):
  kernel_shape = convLayer.getKernelShape()
  height, width, inChannels = kernel_shape[0:3]
  
  stddev = sqrt(2.0 / (height * width * inChannels))
  
  kernel = variableWithWeightDecay('weights', kernel_shape, 
                                   stddev, weightDecay, 
                                   useGetVar, 
                                   device)
  
  return kernel

  
# This might not be correct as I haven't got time to read He et. al.
# carefully  
def heEtAlWeightFFInit(ffLayer,
                       weightDecay,
                       useGetVar = True,
                       device = '/cpu:0'):
  input_size = ffLayer.getInputSize()
  output_size = ffLayer.getOutputSize()
  stddev = sqrt(2.0 / input_size)
  weights = variableWithWeightDecay('weights', [input_size, output_size],
                                    stddev, 
                                    weightDecay,
                                    useGetVar,
                                    device)
  
  return weights       
    
def defaultWDConvInit(convLayer):
    wdInitializer = WeightDecayInitializer(1e-4, 0.0)
    return wdInitializer.weightConvLayerInit(convLayer)

def defaultWDFFInit(ffLayer):
    wdInitializer = WeightDecayInitializer(1.0, 0.0)
    return wdInitializer.weightDecayFFLayerInit(ffLayer)         


def truncatedNormalInit():
    # TODO: put truncated normal here
    pass 
