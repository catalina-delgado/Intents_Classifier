from imports import tf
from src.layers.layers import ConvBlock, AttentionGuided, FullyConnected

class Model():
    def __init__(self, X_train, y_train, **Kwargs):
        super(Model, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        input_shape = self.X_train.shape[1]  # (768) embeddings de BERT
        input_tensor = Input(shape=(input_shape,))
        self.input = Reshape((input_shape, 1))(input_tensor)
    
    def build(self):
        x = ConvBlock(64, 3)(self.input)
        x = AttentionGuided(64)(x)
        x = FullyConnected()(x)
        x = tf.keras.layers.Dense(len(self.y_train[0]), activation='softmax')(x)
        return x
    
    def compile(self):
        model = tf.keras.Model(inputs=self.input, outputs=self.build())
        sgd = SGD(learning_rate=0.005, momentum=0.95, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        return model