from imports import tf, HeNormal

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.conv1D = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', padding='same',  kernel_initializer=HeNormal())
        self.drop = tf.keras.layers.Dropout(0.1)
        self.conv1D_2 = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', padding='same',  kernel_initializer=HeNormal())
        self.normalization = tf.keras.layers.BatchNormalization()
        
    def call(self, input):
        x = self.conv1D(input)
        x = self.drop(x)
        x = self.conv1D_2(x)
        x = self.drop(x)
        x = self.normalization(x)
        return x
    
    def compute_output_shape(self, input_shape):
        conv_output_shape = self.conv1D_2.compute_output_shape(input_shape)
        return conv_output_shape
    

class AttentionGuided(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(AttentionGuided, self).__init__(**kwargs)
        self.units = units
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
        x = tf.expand_dims(x, axis=1)  # Expande las dimensiones para que coincidan con la entrada original
        x = tf.tile(x, [1, tf.shape(input)[1], 1])  # Repite el vector de atención a lo largo de la secuencia
        x = tf.multiply(input, x)
        return x
    
    def compute_output_shape(self, input_shape):
        return (input_shape)
    

class FullyConnected(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FullyConnected, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(32, activation='relu')
        self.drop = tf.keras.layers.Dropout(0.1)
        self.pooling = tf.keras.layers.AveragePooling1D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        
    def call(self, inputs):
        x = self.pooling(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop(x)
        x = self.dense2(x)
        x = self.drop(x)
        x = self.dense3(x)
        x = self.drop(x)
        return x
       