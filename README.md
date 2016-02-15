# Gating-types-for-Residual-Networks
Implement and experiment with different types of gating (preconditioning) for Residual Network on Cifar 10. The gate types are Residual, Highway and a more general version of both Highway and Residual. The code is written in Tensorflow.

I use almost the same configuration as the authors of ResNet used. The errors (%) of each gate type on the test set are as followed:

Gate type | n = 3 (20 layers) | n = 9 (56 layers)
:---: | :---: | :---: |
Residual | 9.3 | 7.8 |
Highway | 9.11 | coming soon
Myway | coming soon | coming soon


The code features convenient wrappers for feedforward, convolution, batchnorm layers and other, less elegant layer types. It can be run on one or multiple GPUs. The code is developed based on the latest Tensorflow source code (not the stable release).


## TODO: 
- More explanation in README
- Refactor and documentation
- Switch back to the default Tensorflow way of handling program arguments


