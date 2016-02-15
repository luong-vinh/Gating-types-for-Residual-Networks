"""
The purpose of this class is to provide a convenient way to create layers
(or more precisely, the output of layers). I could have used functions but 
I found that too be quite awkward for change and the parameter lists just 
keep growing.

A layer will have attributes as operations and inputs to create outputs 
for that layer. The layer object can be used as the output tensor.
Currently, I'm experimenting with duck typing.
"""

import tensorflow as tf
from tensorflow.python.framework import ops

from weight_initialization import xavierFFLayerInit, defaultWDConvInit, \
                                  xavierConvLayerInit

from copy import copy # Shallow copy for speed

import inspect # to do a bit of generic programming here

# TODO: enable visualization by support _activation_summary(layerObject)

# TODO: the code will become messy as the complexity grows. So, I need to refactor 
# from time to time

# TODO: check the way Lasagne, Keras and others do it. My current way is 
# very flexible as each object of a layer can act as a tensor. 

# NOTE: there seems to be some qualitative difference between creating variables
# by tf.get_variable() and tf.Variable(). For example, with tf.get_variable(),
# highway network works fine with the loss = tf.reduce_mean. With tf.Variable(),
# highway network gets stuck (I tried with running from 500 - 2000 mini-batches) 
# with the same loss. With loss = tf.reduce_sum, highway network shows convergence
# with tf.Variable. 

# Learn from class Loss in 
# https://github.com/google/prettytensor/blob/master/prettytensor/pretty_tensor_class.py
# and class tf.Variable

# Should I inherit the Tensor class or just use duck typing for now?
# Will decide this later
# What are the advantages that the object of this class to behave like a tensor object, 
# instead of, say, just calling layer.get_output() for example? It's certainly more
# convenient. Imagine having to call get_output() everywhere a layer object is 
# used
class Layer(object):
    
    # NOTE: any concrete subclass of Layer must have self._output variable
        
    # Now trying to make a layer object behave like a tensor
    # TODO: what kinds of tests I should write to make sure this one work as expected?    
    def _as_graph_element(self):
        return self._output
    
    def _AsTensor(self):
        return self._output
    
    # TODO: do I need value() and ref() methods as in Variable class?
    # Well, just do a simple version for now. Will check later and will
    # add the _snapshot if necessary
    def ref(self):
        return self._output
    def value(self):
        return self._output
    
    def eval(self, session=None):
        return self._output.eval(session = session)
    
    
    # Conversion to tensor. Copy from tf.Variable
    @staticmethod
    def _TensorConversionFunction(layerObj, dtype=None, name=None, as_ref=False):
        """Utility function for converting a Variable to a Tensor."""
        _ = name
        if dtype and not dtype.is_compatible_with(layerObj.dtype):
            raise ValueError(
                "Incompatible type conversion requested to type '%s' for variable "
                "of type '%s'" % (dtype.name, layerObj.dtype.name))
#         if as_ref:
#             return layerObj.ref()
#         else:
#             return layerObj.value()
        return layerObj._output
    
    # Now overload all the operators
    # Copy and modification from Variable
    @staticmethod
    def _RunOp(operator, a, b):
        """Run the operator 'op' for 'a'.
        
        Args:
          operator: string. The operator name.
          a: A Variable.
          b: Second argument to the operator. None if unary.
        Returns:
          The result of the operator.
        """
        # pylint: disable=protected-access
        if b is not None:
            return getattr(ops.Tensor, operator)(a._AsTensor(), b)
        else:
            return getattr(ops.Tensor, operator)(a._AsTensor())
        # pylint: enable=protected-access
    
    @staticmethod
    def _OverloadOperator(operator):
        """Register _RunOp as the implementation of 'operator'.
        
        Args:
          operator: string. The operator name.
        """
        if operator in ["__invert__", "__neg__", "__abs__"]:
            setattr(Layer, operator, lambda a: Layer._RunOp(operator, a, None))
        else:
            setattr(Layer, operator, lambda a, b: Layer._RunOp(operator, a, b))
    # To carry over all overloaded operators from ops.Tensor to Variable, we
    # register the _RunOp() static method as the implementation of all operators.
    # That function dynamically discovers the overloaded operator in ops.Tensor
    # and invokes it after converting the Variable to a tensor.
    @staticmethod
    def _OverloadAllOperators():
        """Register overloads for all operators."""
        for operator in ops.Tensor.OVERLOADABLE_OPERATORS:
            Layer._OverloadOperator(operator)
    # End of copy and modification from Variable
    
    # Gee, a lot of code to emulate Tensor. There must be some way 
    # to automate this procedure when creating a new class emulating Tensor
    @property
    def name(self):        
        return self._output.name

    # Do I need these two?     
#     @property
#     def initializer(self):
#       """The initializer operation for this variable."""
#       return self._initializer_op

#     @property
#     def op(self):
#       """The `Operation` of this variable."""
#       return self._variable.op
    
    @property
    def device(self):
        """The device of this variable."""
        return self._output.device
    
    @property
    def dtype(self):
        """The `DType` of this variable."""
        return self._output.dtype

    
    @property
    def graph(self):
        """The `Graph` of this variable."""
        return self._output.graph
    
    def get_shape(self):
        """The `TensorShape` of this variable.
        
        Returns:
          A `TensorShape`.
        """
        return self._output.get_shape()
 

class ConvLayer(Layer): 
    # Use default argument to make parameters more self-explanatory
    def __init__(self, inputTensor, kernel_shape = [3, 3, 3, 3], 
                 strides = [1, 1, 1, 1],
                 kernelInit = defaultWDConvInit,
                 biasInitVal = 0.0, padding = 'SAME',
                 operator = tf.nn.relu):
        self._kernel_shape = kernel_shape        
        self._biasInitVal = biasInitVal
        self._op = operator
        
        kernel, biases = kernelInit(self)
        conv = tf.nn.conv2d(inputTensor, kernel, strides, padding)
        linear = conv + biases
        if operator is not None:        
            self._output = operator(linear)
        else:
            self._output = linear       

    def getKernelShape(self):
        return self._kernel_shape

    def getBiasInitVal(self):
        return self._biasInitVal
    
    def getOperator(self):
      return self._op       
     


"""
TODO: Check if I can prevent problems with layer._output not being a Tensor
    by using this:
    http://stackoverflow.com/questions/16017397/injecting-function-call-after-init-with-decorator
    
    or just Google for more:
    python post  __init__

TODO: Unfortunately, I run out of time now so I have to make a quick 
design choice here to create a ConvLayer with BatchNormalization. Will  
think about better ways later

TODO: do I need to inherit ConvLayer here? What are the disadvantages?
Do I need to write this class at all? 
"""
class BatchNormedConvLayer(ConvLayer):
  def __init__(self, inputTensor, ewma_trainer, 
               kernel_shape = [3, 3, 3, 3], 
               strides = [1, 1, 1, 1],
               kernelInit = defaultWDConvInit,
               biasInitVal = 0.0, 
               padding = 'SAME',
               operator = tf.nn.relu,
               # Batch norm optional parameters
               scale_after_norm = True,
               epsilon = 0.0001, 
               is_train_phase = True,
               use_scope = False, # TODO: Dirty fix to enable multi-GPU
               scope_name = None):
    
    if use_scope:
      with tf.variable_scope(scope_name + '_conv'):        
        super(self.__class__, self).__init__(
          inputTensor, kernel_shape, strides, kernelInit,
          biasInitVal, padding, None)
    else:                    
      super(self.__class__, self).__init__(
        inputTensor, kernel_shape, strides, kernelInit,
        biasInitVal, padding, None)
    
    # TODO: a major source of bug here
    # TODO: I have to decide how I can deal with this.
    # assignment overwrite is probably possible but is
    # this desirable? Probably    
    # Can't do this for now. I'll have to find out how
    # I can overwrite assignment here
#     self._output = BatchNormLayer(self._output,
#                     kernel_shape[3], ewma_trainer,
#                     scale_after_norm, epsilon,
#                     is_train_phase)    
    # this is not really nice    
    if use_scope:
      with tf.variable_scope(scope_name + '_batch_norm'):
        self._output = BatchNormLayer(self._output,
                    kernel_shape[3], ewma_trainer,
                    scale_after_norm, epsilon,
                    is_train_phase) * 1
    else:
      self._output = BatchNormLayer(self._output,
                    kernel_shape[3], ewma_trainer,
                    scale_after_norm, epsilon,
                    is_train_phase) * 1     
    
    if operator is not None:
      self._output = operator(self._output)                            

    
    
# Fully connected aka feed forward layer
# inputTensor of shape: batch_size x input_size
# The output of this layer (self._output) is of shape: batch_size x output_size
class FFLayer(Layer):
    def __init__(self, inputTensor, output_size, biasInitVal = 0.0,
             weightInit = xavierFFLayerInit, 
             operator = tf.nn.relu, name = None):     
        
        # Do I actually need to store the operator, weight and bias?        
        # Maybe operator can be useful in emulating tensor
        self._input_size = inputTensor.get_shape().as_list()[1]
        self._operator = operator
        self._output_size = output_size
        self._biasInitVal = biasInitVal        
              
        weight, bias = weightInit(self)        
        
        linear = tf.matmul(inputTensor, weight) + bias        
        
        if self._operator is not None:
            self._output = self._operator(linear)
        else:
            self._output = linear
    
        # The Java way       
    def getInputSize(self):
        return self._input_size
    def getOutputSize(self):
        return self._output_size    
    def getOperator(self):
        return self._operator
    def getBiasInitVal(self):
        return self._biasInitVal   

# A layer for feedforward highway layer, residual layer and the 
# more general version of them. 
# Should I write a general class that deals with both FFLayer and ConvLayer?
# No, probably it's not worth the trouble  
class MywayFFLayer(Layer):
    # Maybe I should use enum class if that exists in Python
    # TODO: find better way to create constants
    HIGHWAY_GATE = 13113
    RESIDUAL_GATE = 81382
    MYWAY_GATE = 37802    
    
    # second input's type:
    SAME = 70831 
    NONLINEAR = 40982
    # TODO: support for padding and subsampling for dimension matching 
    
    # TODO: add biasGate
    # In Myway layer, HighwayLayer and Residual layer, input size is assumed 
    # to be the same as output size. If not, just do a linear transformation
    # before calling this layer.
    # TODO: This assumption limits the generality of this class. Will remove this 
    # assumption and allow options of padding, subsampling and linear/nonlinear 
    # transform
    # I just need to add output_size parameter
    def __init__(self, inputTensor, biasInitVal = 0.0, weightInit = xavierFFLayerInit, 
                 operator = tf.nn.relu, name = None, gateType = HIGHWAY_GATE,
                 secondInputType = SAME):
        
        output_size = inputTensor.get_shape().as_list()[1]
        # Probably, there's no need to call super
#         super(self.__class__, self).__init__(inputTensor, output_size, 
#                biasInitVal, #weightInit,
#                operator, name)         
        # Use default for now
        Hx = FFLayer(inputTensor, output_size)
        # transform gate
        Tx = FFLayer(inputTensor, output_size, biasInitVal = -2.0, operator = tf.sigmoid)
        
        Gx = inputTensor
        Cx = 1 # Residual gate
        # OK, instead of dictionary-base switch case, I'll just do the good old ifelse here        
        if gateType == MywayFFLayer.HIGHWAY_GATE:
            # Carry gate
            #print("Create highway gate")
            Cx = 1 - Tx        
        elif gateType == MywayFFLayer.MYWAY_GATE: # General version for carry gate
            Cx = FFLayer(inputTensor, output_size, biasInitVal = -2.0, operator = tf.sigmoid)
        
        # TODO: for residual gate, T(x) = 1 as well. Although, I don't need T(x) = 1 here
        # to make the network converge         
            
        if secondInputType == MywayFFLayer.NONLINEAR:
            Gx = FFLayer(inputTensor, output_size)
        
        self._output = Tx * Hx + Cx * Gx         


# TODO: This will be very similar to MyWayFFLayer. Check if I can improve by refactoring
# the common code and/or the design
# -----> check the code of the function createStackedLayer below
# TODO: Use MyWayFFLayer's constants for now. Need to find better ways
# TODO: Support options for padding 0 or subsampling to match shapes between
# inputTensor and the outputs of the convolution transformation
# Follow tensorflow's convention: 
# input_tensor: batch x height x width x numChannels
# kernel_shape: height x width x numInputFilters x numOutputFilters
# This class is a shallow version of the class StackedMyWayLayer below 
# but it doesn't have the restrictions of that class
class MyWayConvLayer(Layer):
    # kernel_shape: 
    def __init__(self, inputTensor, kernel_shape, strides, blockBiasVal = 0.0,
                 gateBiasVal = -2.0, kernelInit = defaultWDConvInit, 
                 operator = tf.nn.relu, name = None, 
                 gateType = MywayFFLayer.HIGHWAY_GATE,
                 secondInputType = MywayFFLayer.SAME):
        # The same procedure as for MyWayFFLayer. Only the classes and parameters 
        # are different. There are probably ways to refactor this        
        Hx = ConvLayer(inputTensor, kernel_shape = kernel_shape, strides = strides,
                       kernelInit = kernelInit, biasInitVal = blockBiasVal,
                       operator = operator)
        
        # transform gate
        Tx = ConvLayer(inputTensor, kernel_shape = kernel_shape, strides = strides,
                       kernelInit = kernelInit, biasInitVal = gateBiasVal,
                       operator = tf.sigmoid)        
        
        Gx = inputTensor
        # Check if numChannels matches with numOutputFilters
        numChannels = inputTensor.get_shape().as_list()[3]
        numOutputFilters = kernel_shape[3]
        
        if secondInputType != MywayFFLayer.SAME:
            Gx = ConvLayer(inputTensor, kernel_shape = kernel_shape, strides = strides,
                       kernelInit = kernelInit, biasInitVal = blockBiasVal,
                       operator = operator)
        elif numChannels != numOutputFilters:
            # Use linear convolution for now
            Gx = ConvLayer(inputTensor, kernel_shape = kernel_shape, strides = strides,
                       kernelInit = kernelInit, biasInitVal = 0.0,
                       operator = None)
        Cx = 1  # Residual gate
        if gateType == MywayFFLayer.HIGHWAY_GATE:
            Cx = 1 - Tx
        elif gateType == MywayFFLayer.MYWAY_GATE: # General version for carry gate
            Cx = ConvLayer(inputTensor, kernel_shape = kernel_shape, strides = strides,
                       kernelInit = kernelInit, biasInitVal = gateBiasVal,
                       operator = tf.sigmoid)             
        
        self._output = Tx * Hx + Cx * Gx

  
# TODO: write a lesson on trying to make code more generic. It's very 
# troublesome, has some hard code and assumptions and probably not worth 
# the effort. I could have done better but then I'd have to spend more 
# time for that. The time cost is a bit too much 

# TODO: I'm not satisfied with these Stack* classes. Will refactor the design
# and the code later. The difficulty of change to insert
# ConvBatchNormLayer to these class is probably the sign of many 
# things going wrong with this design.
# Or I just get rid of these classes and write them as functions.
# Probably get rid of the generic stuffs as well

# TODO: not-so-nice fix to enable the choice between tf.Variable 
# and tf.get_variable. It's not easy to find better solutions 

"""
This class is an generic extension of MyWayConvLayer and MyWayFFLayer. 
It supports an option of allowing how many 
layers to be stacked before doing the multiplicative gating.

Assume the creation of each layer here uses the same hyperparameters 
(same kernel shape, weight shape etc..). The only different parameter is 
the input of each layer. This is a little bit limited
but as I want to create some kind of generic function for both types 
of feedforward and convolution layers, I have to make some restrictions.
I can get rid of this restriction but then it'll create other 
assumptions 

TODO: make sure it works correctly. I work on this when I'm very tired 
and distressed
"""
class StackedMyWayLayer(Layer):
  def __init__(self, *args, **kwargs):
    allParams = locals()['kwargs']
    #self.allParams = allParams  # debug
    # haven't got away from hard-coded :(
    inputTensor = allParams['inputTensor']
    layerType = allParams['layerType']
    gateBiasVal = allParams['gateBiasVal']
    gateType = allParams['gateType']
    secondInputType = allParams['secondInputType']
    numStackedLayers = allParams['numStackedLayers']
      
    paramDict = {}    
    argSpecs = inspect.getargspec(layerType.__init__)
    for paramKey in argSpecs.args[1:]: # ignore the first parameter "self" 
      paramDict[paramKey] = allParams[paramKey]   

    
    self._output = self.compute(inputTensor, layerType, paramDict, 
           gateBiasVal, gateType,
           secondInputType,  
           numStackedLayers)
   
   
  
  def compute(self, inputTensor, layerType, paramDict, 
             gateBiasVal = -2.0,                 
             gateType = MywayFFLayer.HIGHWAY_GATE,
             secondInputType = MywayFFLayer.SAME, 
             numStackedLayers = 2):
    
    operator = paramDict['operator'] # Note another assumption here
    
    use_scope = False    
    if paramDict.has_key('use_scope'):
      use_scope = paramDict['use_scope']      
    
    # Note that the last layer of the output of createStackedLayer
    # is a linear layer  
    if use_scope:
      paramDict['scope_name'] += '_first_input'
    Hx = self.createStackedLayer(layerType, paramDict, numStackedLayers)
    
    # ResNet uses linear addition before applying nonlinear operator. This 
    # is the only difference between ResNet and MywayNet
    # TODO: create more variants of MywayNet with multiplicative addition 
    # before/after applying nonlinear op  
    if gateType != MywayFFLayer.RESIDUAL_GATE:
      Hx = operator(Hx)
    
    Gx = inputTensor    
    # No need for this now due to the restriction of createStackedLayer
    # Check if numInputChannels matches with numOutputFilters
#     numInputChannels = inputTensor.get_shape().as_list()[3]
#     numOutputFilters = kernel_shape[3]
        
    if secondInputType != MywayFFLayer.SAME:
#         Gx = ConvLayer(inputTensor, kernel_shape = kernel_shape, strides = strides,
#                    kernelInit = kernelInit, biasInitVal = blockBiasVal,
#                    operator = operator)    
      if use_scope:
        paramDict['scope_name'] += '_second_input'
      # I made mistake here. I only need to create a stack of layers for Hx
      # For the rest, one layer is enough  
      #Gx = self.createStackedLayer(layerType, paramDict, numStackedLayers)      
      Gx = self.createStackedLayer(layerType, paramDict, numStackedLayers = 1)
      Gx = operator(Gx)  
        
   
    # Set parameters for gate
    # Set bias gate -- assumption: the same key is used by all
    paramDict['biasInitVal'] = gateBiasVal   
    paramDict['operator'] = tf.sigmoid
    # This will break the generic feature of this class but I will rewrite this 
    # class later anyway
    # Should have called kernelInit as weightInit, well never mind
    paramDict['kernelInit'] = xavierConvLayerInit
    
    # TODO: add parameter for gate operator. Experiment with tanh, relu    
    Tx = 1  # Residual type
    if gateType != MywayFFLayer.RESIDUAL_GATE:
      if use_scope:
        paramDict['scope_name'] += '_transform_gate'
      #Tx = self.createStackedLayer(layerType, paramDict, numStackedLayers)
      # Still need sigmoid here because the output of createStackLayer is linear      
      Tx = self.createStackedLayer(layerType, paramDict, numStackedLayers = 1)
      Tx = tf.sigmoid(Tx)
    
    paramDict['operator'] = tf.sigmoid
    # Carry gate
    Cx = 1  # Residual type
    if gateType == MywayFFLayer.HIGHWAY_GATE:
      Cx = 1 - Tx
    elif gateType == MywayFFLayer.MYWAY_GATE: # General version for carry gate
      if use_scope:
        paramDict['scope_name'] += '_carry_gate'
      #Cx = self.createStackedLayer(layerType, paramDict, numStackedLayers)
      Cx = self.createStackedLayer(layerType, paramDict, numStackedLayers = 1)
      Cx = tf.sigmoid(Cx)     
        
    output = Tx * Hx + Cx * Gx
    
    if gateType == MywayFFLayer.RESIDUAL_GATE:
      # apply nonlinear op here
      # See the TODO above      
      output = operator(output)
          
    return output
  
  #   Using unpacking arguments to make a common interface for 2 classes
  #   curious: What is the performance of unpacking arguments in Python? 
  #   Assume each layer here uses the same parameters. The only different
  #   parameter is the input of each layer. This is a little bit limited
  #   but as I want to create some kind of generic function for both types 
  #   of feedforward and convolution layers, I have to make some restrictions.
  #   I can get rid of this restriction but then it'll create other 
  #   assumptions.
  #   In hindsight, this type of generic layer creation makes it harder to
  #   understand 
  def createStackedLayer(self, layerType, paramDict,                          
                         numStackedLayers = 2):
                        # scope_name = None):
    # Just need to deal with the case numStackedLayers = 1
    out = paramDict["inputTensor"] 
    
    use_scope = False
    if paramDict.has_key('use_scope'):
      use_scope = paramDict['use_scope']
    
    # Don't include the last layer here because the last layer will be linear   
    for layerIndex in xrange(numStackedLayers - 1):
      if use_scope:
        currentScopeName = paramDict['scope_name'] 
        paramDict['scope_name'] = currentScopeName + \
                                  '_' + str(layerIndex)       
      out = layerType(**paramDict)
      
      # Assume the input argument for each layer's constructor has the 
      # name as "inputTensor". There might be better ways to do this 
      # but I don't have time to find out now
      paramDict["inputTensor"] = out 
    
    # the last layer is linear
    # Another assumption  that the argument name for the operation in each
    # layer type is "operator"
    if use_scope:
      currentScopeName = paramDict['scope_name'] 
      paramDict['scope_name'] = currentScopeName + '_linear_ouput'    
    
    paramDict["operator"] = None
    out = layerType(**paramDict)
      
    return out

"""
  These convenient classes below must have the constructor matched with the constructor of
  ConvLayer with the addition of other parameters to support the construction
  of a stacked Myway layer.
  They really just different instances of StackedMyWayLayer objects. 
  Follow tensorflow's convention: 
  input_tensor: batch x height x width x numChannels
  kernel_shape: height x width x numInputFilters x numOutputFilters
  due to the restriction in the createStackedLayer method, numChannels, 
  numInputFilters, numOutputFilters must have the same value."""
class StackedMyWayConvLayer(StackedMyWayLayer):
  def __init__(self, inputTensor, kernel_shape, strides,
             kernelInit = defaultWDConvInit, 
             biasInitVal = 0.0, padding = 'SAME',
             operator = tf.nn.relu, # End of ConvLayer arguments
             gateBiasVal = -2.0,                              
             gateType = MywayFFLayer.HIGHWAY_GATE,
             secondInputType = MywayFFLayer.SAME,              
             layerType = ConvLayer,
             numStackedLayers = 2): # Follow Residual network's convention
    allParams = copy(locals())
    # Must remove the 'self' key, otherwise I'll get into problem with 
    # the super class init
    allParams.pop('self', None) 
    super(self.__class__, self).__init__(**allParams)

# A quick (and probably dirty) class for stacked, batch-normed conv layer
# TODO: look ugly. Will find better ways later (Even doing this manually would
# look cleaner)
class StackedBatchNormedConvLayer(StackedMyWayLayer):
  def __init__(self, inputTensor, ewma_trainer, 
           kernel_shape, strides = [1, 1, 1, 1],
           kernelInit = defaultWDConvInit, 
           biasInitVal = 0.0, padding = 'SAME',
           operator = tf.nn.relu, # End of ConvLayer arguments
           gateBiasVal = -2.0,                              
           gateType = MywayFFLayer.HIGHWAY_GATE,
           secondInputType = MywayFFLayer.SAME,              
           layerType = BatchNormedConvLayer,
           numStackedLayers = 2, # Follow Residual network's convention
           # Batch norm optional parameters
           scale_after_norm = True,
           epsilon = 0.0001, 
           is_train_phase = True,           
           use_scope = False,
           scope_name = None):
    
    allParams = copy(locals())
    # Must remove the 'self' key, otherwise I'll get into problem with 
    # the super class init
    allParams.pop('self', None) 
    super(self.__class__, self).__init__(**allParams)


"""
  Essentially, dga's code with 3 changes. First, make this class inherit
  Layer class. Second, do the saving of the moving average after the calculation
  of the mean and variance but before the normalization occurs (bgshi's idea).
  Third, do the restore work outside the class, not inside as it doesn't work
  with restoring a session from file
  --> UPDATE: I also move all the moving average computation outside because 
  probably there are some Tensorflow bugs when doing so under tf.variable_scope
  (the saved names of the moving averages are wrong in that case) 
  TODO: This is currently written for convolution layer, will make it work for
  feedforward layer later.
  
  TODO: try replacing the if-else part here symbolic condition (learned from 
  bgshi's code but note that his/her code has some obvious problems). 
  While using symbolic condition may make the graph bigger (probably not
  that much in this case), it makes the reuse of the graph in case of training
  and testing in the same session much easier. 
  
  TODO: write example of computing and restoring the mean and variance's moving
  average outside this class
  
  TODO: get rid of expMOvAvg parameter in the constructor and in the related
  class   
"""
class BatchNormLayer(Layer):
  
  # Collection ID to store the the mean and the variance. 
  # This will make sure we can restore the mean and variance 
  # outside this class and thus make it work even
  # in the case of restoring a training session before doing inference
  # Just generate a random UUID by uuid4()
  batchNormCollectionID = 'bd13581d-6917-4bb0-8f83-b45e26ef48c2'
  
  """Helper class that groups the normalization logic and variables.      

  Use:                                                                      
      ewma = tf.train.ExponentialMovingAverage(decay=0.99)                  
      x = BatchNormLayer(inputTensor, depth, epsilon, ewma,
                            scale_after_norm, is_train_phase)
      (the output x will be batch-normalized).                              
  """

  def __init__(self, inputTensor, depth, expoMovAvg, 
               scale_after_norm = True,
               epsilon = 0.0001, 
               is_train_phase = True):
#     self.mean = tf.Variable(tf.constant(0.0, shape=[depth]),
#                             trainable=False, name = 'mean')
#     self.variance = tf.Variable(tf.constant(1.0, shape=[depth]),
#                                 trainable=False, name='variance')
#     self.beta = tf.Variable(tf.constant(0.0, shape=[depth]), name = 'beta')
#     self.gamma = tf.Variable(tf.constant(1.0, shape=[depth]), name = 'gamma')
      # Just a quick fix. Will do it properly later
    # Putting device '/cpu:0' here  creates an error "AttrValue must not 
    # have reference type value of float_ref" for variance. I will find 
    # out why later  
    #with tf.device('/cpu:0'):
    # Assign the variable to collection when creating it to avoid 
    # adding duplicates        
    self.mean = tf.get_variable('mean', [depth], #tf.float32,
                        initializer = tf.constant_initializer(0.0),
                        trainable = False, 
                        collections = [BatchNormLayer.batchNormCollectionID])
    self.variance = tf.get_variable('variance', [depth], #tf.float32,
                        initializer = tf.constant_initializer(1.0),
                        trainable = False,
                        collections = [BatchNormLayer.batchNormCollectionID])    
    
#     tf.add_to_collection(BatchNormLayer.batchNormCollectionID, self.mean)
#     tf.add_to_collection(BatchNormLayer.batchNormCollectionID, self.variance)
    
    self.beta = tf.get_variable('beta', [depth], #tf.float32,
                                initializer = tf.constant_initializer(0.0))
    self.gamma = tf.get_variable('gamma', [depth], # tf.float32,
                                initializer = tf.constant_initializer(1.0))
    
      
    #self.expoMovAvg = expoMovAvg
#     self.epsilon = epsilon
#     self.scale_after_norm = scale_after_norm
    if is_train_phase:
      # These indirection assignments are probably not necessary anymore. I'll
      # check that later when I got time
      mean, variance = tf.nn.moments(inputTensor, [0, 1, 2])
      assign_mean = self.mean.assign(mean)
      assign_variance = self.variance.assign(variance)
      
      # TODO: Is saving outside batch norm a cleaner solution? 
      # I have no idea why creating this op under a variable scope make Saver 
      # object stored the wrong moving averages     
      # Yeah, I'll save them outside
      #saveMovAveOp = expoMovAvg.apply([self.mean, self.variance])
      
      with tf.control_dependencies([assign_mean, assign_variance]):
        #with tf.control_dependencies([saveMovAveOp]):
#           apparently, tf.nn.batch_norm_with_global_normalization 
#           changes its mean and variance parameters. Because in my previous 
#           implementation, I pass BatchNormLayer's mean and variance members 
#           directly to that function, instead of passing a copy as dga did,  
#           so my result is different from dga's result

          self._output = tf.nn.batch_norm_with_global_normalization(
              inputTensor, mean, variance, 
              self.beta, self.gamma,
              epsilon, scale_after_norm)
    else: # Restore saved moving average of mean and variance, which 
      # are approximation of the population mean and variance
      # Unfortunately, this only works if we run training and testing
      # in the same session. It doesn't work in case we restore 
      # a session from a saved checkpoint file. For now, I assume the
      # restoration work will occur somewhere outside this class. It 
      # will happen when the model's graph is completed
      #mean = expoMovAvg.average(self.mean)
      #variance = expoMovAvg.average(self.variance)        
         
      # This is probably not necessary, according to dga
      #local_beta = tf.identity(self.beta)
      #local_gamma = tf.identity(self.gamma)
        
      # Add to collection to restore after graph building
      # Use the variable name for now
        
      # Well, debug a bit
      # Copy in order to pass the copies to batch_norm_with_global_normalization      
#       self.mean_clone = tf.identity(self.mean)
#       self.variance_clone = tf.identity(self.variance)
      mean = tf.identity(self.mean)
      variance = tf.identity(self.variance)
            
      self._output = tf.nn.batch_norm_with_global_normalization(
          inputTensor, mean, variance, self.beta, self.gamma,
          epsilon, scale_after_norm) 
           
    # Finally, add the names of the mean and variance to 
    # a global batch norm collection so that we can save and restore
    # moving average outside this BatchNorm class. For some weird reasons, 
    # when an expoMovAvg saves variables under "with tf.variable_scope", 
    # it saves their moving averages under the wrong names 
#     tf.add_to_collection(BatchNormLayer.batchNormCollectionID, 
#                            #expoMovAvg.average_name(self.mean))
#                            self.mean.name) 
#     tf.add_to_collection(BatchNormLayer.batchNormCollectionID, 
#                            #expoMovAvg.average_name(self.variance))
#                            self.variance.name) 
 
"""
Because of the restrictions of the class StackedMyWayLayer (so that
it can work with both feedforward and convolution layers), it's very
difficult to create that layer in the case where the input tensor's
shape doesn't match with the ouput tensor's. I run out of time so, for now,
I just create a function to address the case of Conv here. Will find better 
ways later
"""
# Assume kernel_shape[2] == kernel_shape[3], that is, when using 
# kernel shape, the number of input channels must be the same as that
# of output channels
def createStackedConvLayer(inputTensor, ewma_trainer, 
                           kernel_shape, strides = [1, 2, 2, 1],
                           kernelInit = defaultWDConvInit, 
                           biasInitVal = 0.0, padding = 'SAME',
                           operator = tf.nn.relu, 
                           gateBiasVal = -2.0,                 
                           gateType = MywayFFLayer.HIGHWAY_GATE,
                           secondInputType = MywayFFLayer.SAME, 
                           numStackedLayers = 2, 
                           # Batch norm optional parameters
                           scale_after_norm = True,
                           epsilon = 0.0001, 
                           is_train_phase = True,
                           use_scope = False, # TODO: Dirty fix to enable multi-GPU
                           scope_name = None):
  
  # First transform input tensor to match with the number 
  # of output channels. Change kernel shape to do mapping
  # by convolution 
  numInputChannels = inputTensor.get_shape().as_list()[3]  
  numKernelInputChannels = kernel_shape[2]
  kernel_shape[2] = numInputChannels  
  
  commonScope = None
   
  
  if use_scope:
    commonScope = scope_name + '_first_input'
  # Input for shortcut connection
  # Because the assumption is that inputTensor's shape is different so
  # I need to linearly transform the inputTensor to match its dimension to that
  # of the gates.
  linearOutput = BatchNormedConvLayer(inputTensor, ewma_trainer,
                               kernel_shape, strides,
                               kernelInit, biasInitVal,
                               padding, None,
                               scale_after_norm, epsilon,
                               is_train_phase,
                               use_scope,
                               commonScope)
  
  Gx = linearOutput
  if secondInputType != MywayFFLayer.SAME:
    # Use nonlinear, instead of linear output
    Gx = operator(Gx)
  
  # Transform gate
  Tx = 1
  if use_scope:
    commonScope = scope_name + '_transform_gate'
  if gateType != MywayFFLayer.RESIDUAL_GATE:
    Tx = BatchNormedConvLayer(inputTensor, ewma_trainer,
                              kernel_shape, strides,
                              kernelInit, biasInitVal,
                              padding, tf.nn.sigmoid,
                              scale_after_norm, epsilon,
                              is_train_phase, 
                              use_scope,
                              commonScope)
  
  # Carry gate
  if use_scope:
    commonScope = scope_name + '_carry_gate'
  Cx = 1  # Residual type
  if gateType == MywayFFLayer.HIGHWAY_GATE:
    Cx = 1 - Tx
  elif gateType == MywayFFLayer.MYWAY_GATE: # General version for carry gate
    Cx = BatchNormedConvLayer(inputTensor, ewma_trainer,
                              kernel_shape, strides,
                              kernelInit, biasInitVal,
                              padding, tf.nn.sigmoid,
                              scale_after_norm, epsilon,
                              is_train_phase,
                              use_scope,
                              commonScope)   
  
  if use_scope:
    commonScope = scope_name + '_shortcut_input'  
  # Inputs for shortcut blocks
  blockInput = BatchNormedConvLayer(inputTensor, ewma_trainer,
                               kernel_shape, strides,
                               kernelInit, biasInitVal,
                               padding, operator,
                               scale_after_norm, epsilon,
                               is_train_phase,
                               use_scope,
                               commonScope)
  
  
  # Restore the original kernel_shape
  kernel_shape[2] = numKernelInputChannels
  
  for layerIndex in xrange(numStackedLayers - 1):
    if use_scope:
      commonScope = scope_name + '_shortcut_layer_' + str(layerIndex) 
    blockInput = BatchNormedConvLayer(blockInput, ewma_trainer,
                               kernel_shape, [1, 1, 1, 1],
                               kernelInit, biasInitVal,
                               'SAME', operator,
                               scale_after_norm, epsilon,
                               is_train_phase,
                               use_scope,
                               commonScope)
    
  Hx = blockInput  
  
  output = Tx * Hx + Cx * Gx
  
  if gateType == MywayFFLayer.RESIDUAL_GATE:
    # apply nonlinear op here
    # See the TODO above      
    output = operator(output)
        
  return output        

# pylint: disable=protected-access
ops.register_tensor_conversion_function(Layer,
                                        Layer._TensorConversionFunction)
Layer._OverloadAllOperators()
# pylint: enable=protected-access  
    
