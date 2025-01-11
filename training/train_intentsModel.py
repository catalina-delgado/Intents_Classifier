from functions import NeuroStage
from imports import tf, np
from src.models.model import Model

class TrainModel(NeuroStage):
    def __init__(self, batch_size=32, epochs=4, model_name='', models=None):
        super().__init__()
        self.BATCH_SIZE = batch_size
        self.EPHOCS = epochs
        self.MODEL_NAME = model_name
       
    def train(self):
        X_train = np.random.rand(100, 256, 256, 1)
        y_train = np.random.randint(0, 2, 100) 
        X_val = np.random.rand(20, 256, 256, 1) 
        y_val = np.random.randint(0, 2, 20)
        
        self.architecture = Model(X_train, y_train)
        self.model = self.architecture.compile()
        
        self.init_fit(self.model, X_train, y_train, X_val, y_val, self.EPHOCS, self.BATCH_SIZE, self.MODEL_NAME)