from functions import NeuroStage
from imports import tf, np
import os
from src.models.model import Model

class TrainModel(NeuroStage):
    def __init__(self, batch_size=32, epochs=4, model_name='', models=None):
        super().__init__()
        self.BATCH_SIZE = batch_size
        self.EPHOCS = epochs
        self.MODEL_NAME = model_name
       
    def train(self):
        print(os.getcwd())
        X_train = np.load('data/X_train.npy')
        y_train = np.load('data/y_train.npy')
        X_val = np.load('data/X_val.npy')
        y_val = np.load('data/y_val.npy')
        
        self.architecture = Model(X_train, y_train)
        self.model = self.architecture.compile()
        
        self.init_fit(self.model, X_train, y_train, X_val, y_val, self.EPHOCS, self.BATCH_SIZE, self.MODEL_NAME)