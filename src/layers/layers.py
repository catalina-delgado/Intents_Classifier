from imports import tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(ConvBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.conv1D = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', padding='same',  kernel_initializer=HeNormal())
        self.drop = tf.keras.layers.Dropout(0.1)
        self.normalization = tf.keras.layers.BatchNormalization()
        
    def call(self, input):
        x = self.conv1D(input)
        x = self.drop(x)
        x = self.conv1D(x)
        x = self.drop(x)
        x = self.normalization(x)
        return x
    

class AttentionGuided(tf.keras.layers.Layer):
    def __init__(self, units):
        super(ConvBlock, self).__init__()
        self.pooling = tf.keras.layers.GlobalMaxPooling1D()
        self.denseRelu = tf.keras.layers.Dense(units, activation='relu')
        self.drop = tf.keras.layers.Dropout(0.1)
        self.denseSigmoid = tf.keras.layers.Dense(units, activation='sigmoid')
        
    def call(self, input):
        x = self.pooling(input)
        x = self.drop(x)
        x = self.denseRelu(x)
        x = self.drop(x)
        x = self.denseSigmoid(x)
        x = tf.keras.multiply([input, x])
        return x
    

class FullyConnected(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FullyConnected, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(32, activation='relu')
        self.drop = tf.keras.layers.Dropout(0.1)
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.drop(x)
        x = self.dense2(x)
        x = self.drop(x)
        x = self.dense3(x)
        x = self.drop(x)
        return x
       