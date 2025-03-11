import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import logging
import os
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class XGBoostModel:
    """
    Classe pour le modèle XGBoost utilisé pour prédire les retards de trains.
    """
    def __init__(self, params=None):
        """
        Initialise le modèle XGBoost avec des paramètres optionnels.
        
        Args:
            params (dict, optional): Paramètres du modèle XGBoost.
        """
        self.model = None
        self.default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 10,
            'learning_rate': 0.04,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.05,
            'reg_lambda': 1.0,
            'eval_metric': 'mae'
        }
        self.params = params if params is not None else self.default_params
        
    def train(self, X_train, y_train, X_val=None, y_val=None, num_boost_round=350, early_stopping_rounds=20):
        """
        Entraîne le modèle XGBoost.
        
        Args:
            X_train: Features d'entraînement
            y_train: Cible d'entraînement
            X_val: Features de validation (optionnel)
            y_val: Cible de validation (optionnel)
            num_boost_round: Nombre d'itérations d'entraînement
            early_stopping_rounds: Nombre d'itérations sans amélioration avant d'arrêter
        
        Returns:
            self: L'instance du modèle
        """
        logger.info("Création des matrices DMatrix pour XGBoost...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evallist = [(dtrain, 'train'), (dval, 'validation')]
        else:
            evallist = [(dtrain, 'train')]
            
        logger.info("Entraînement du modèle XGBoost...")
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evallist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=10
        )
        
        return self
    
    def predict(self, X):
        """
        Effectue des prédictions avec le modèle entraîné.
        
        Args:
            X: Features pour la prédiction
            
        Returns:
            array: Prédictions
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez d'abord la méthode train().")
            
        dtest = xgb.DMatrix