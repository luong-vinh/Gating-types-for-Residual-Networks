# Gating-types-for-Residual-Networks
Implement and experiment with different types of gating (preconditioning) for Residual Network on Cifar 10. The gate types are Residual, Highway and a more general version of both Highway and Residual. The code is written in Tensorflow.
Basically, in ResNet, a residual layer, H(x) + x, is a particular instance of a simple, more general version of Highway network, which I call "Myway" (mainly because of the English version of the French song) in order to distinguish between the particular version used in the Highway Network paper and the more general version implemented here. 
The system currently implements 3 types of gate: 
- Residual: H(x) + x
- Highway: T(x) * H(x) + (1 - T(x)) * x 
- Myway: T(x) * H(x) + C(x) * x.

I use almost the same configuration as the authors of ResNet used. However, in the case of 56-layer network, Highway and Myway gates' losses become very big initially with momentum. I have to switch to ADAM instead.  The errors (%) of each gate type on the test set are as followed:

Gate type | n = 3 (20 layers) | n = 9 (56 layers)
:---: | :---: | :---: |
Residual | 9.3 | 7.8 |
Highway | 9.11 | 8.9|
Myway | 9.3 | 8.4|



The code features convenient wrappers for feedforward, convolution, batchnorm layers and other, less elegant layer types. It can be run on one or multiple GPUs. I used the latest Tensorflow source code (not the stable release).


## TODO: 
- Refactor and documentation: Currently being refactored in my ongoing project
- Switch back to the default Tensorflow way of handling program arguments
- Clean up and upload test code

## Acknowledgement:
The code is developed and tested using a Titan Z card generously donated to me by NVIDIA.
