# Because tf.app.flags seems limited in passing relevant configurations
# I use a quick way here
# One way to pass configuration between this model file and the files which
# actually run this model for different datasets. Use this for now, will 
# find better ways when having time
config = {}
def getConfig():
    return config

# So, most of time, the control file will set a network configuration here,
# then the model file will take the newly set configuration. Probably could
# have done better using a class
def setConfig(externalConfig):
    config = externalConfig

##### Maybe I can define a standard, default configuration set here 
