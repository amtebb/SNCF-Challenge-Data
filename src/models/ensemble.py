import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import logging
import os
import json
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnsembleModel:
    """
    Classe pour combiner plusieurs modèles dans un ensemble.
    """
    def __init__(self, models=None, weights=None):
        """
        Initialise l'ensemble de modèles.
        
        Args:
            models (list, optional): Liste des modèles à inclure dans l'ensemble
            weights (list, optional): Poids pour chaque modèle
        """
        self.models = models if models is not None else []
        self.weights = weights if weights is not None else []
        
        # Si les poids ne sont pas fournis, on utilise des poids égaux
        if len(self.weights) == 0 and len(self.models) > 0:
            self.weights = [1.0 / len(self.models)] * len(self.models)
            
    def add_model(self, model, weight=None):
        """
        Ajoute un modèle à l'ensemble.
        
        Args:
            model: Modèle à ajouter
            weight (float, optional): Poids du modèle
        """
        self.models.append(model)
        
        # Calculer le poids si non fourni
        if weight is None:
            # Normalisation des poids pour qu'ils somment à 1
            n_models = len(self.models)
            self.weights = [1.0 / n_models] * n_models
        else:
            self.weights.append(weight)
            # Normalisation des poids
            sum_weights = sum(self.weights)
            self.weights = [w / sum_weights for w in self.weights]
            
        logger.info(f"Modèle ajouté à l'ensemble. Nombre total de modèles: {len(self.models)}")
        logger.info(f"Poids actuels: {self.weights}")
        
    def predict(self, X):
        """
        Effectue des prédictions en combinant les prédictions de tous les modèles.
        
        Args:
            X: Features pour la prédiction
            
        Returns:
            array: Prédictions pondérées
        """
        if len(self.models) == 0:
            raise ValueError("Aucun modèle n'a été ajouté à l'ensemble.")
            
        predictions = []
        
        # Obtenir les prédictions de chaque modèle
        for model in self.models:
            model_preds = model.predict(X)
            predictions.append(model_preds)
            
        # Combiner les prédictions avec les poids
        weighted_preds = np.zeros_like(predictions[0])
        for i, preds in enumerate(predictions):
            weighted_preds += preds * self.weights[i]
            
        return weighted_preds
    
    def evaluate(self, X, y):
        """
        Évalue l'ensemble sur un jeu de données.
        
        Args:
            X: Features pour l'évaluation
            y: Cible réelle
            
        Returns:
            float: Erreur absolue moyenne (MAE)
        """
        predictions = self.predict(X)
        mae = mean_absolute_error(y, predictions)
        logger.info(f"MAE de l'ensemble sur l'évaluation: {mae:.4f}")
        return mae
    
    def optimize_weights(self, X_val, y_val, n_trials=100):
        """
        Optimise les poids de l'ensemble sur l'ensemble de validation.
        
        Args:
            X_val: Features de validation
            y_val: Cible de validation
            n_trials: Nombre d'essais pour l'optimisation
            
        Returns:
            list: Poids optimisés
        """
        if len(self.models) == 0:
            raise ValueError("Aucun modèle n'a été ajouté à l'ensemble.")
            
        logger.info("Optimisation des poids de l'ensemble...")
        
        # Obtenir les prédictions de base de chaque modèle
        base_predictions = []
        for model in self.models:
            model_preds = model.predict(X_val)
            base_predictions.append(model_preds)
            
        # Fonction pour calculer la MAE avec des poids donnés
        def calculate_mae(weights):
            # Normaliser les poids
            sum_weights = sum(weights)
            norm_weights = [w / sum_weights for w in weights]
            
            # Calculer les prédictions pondérées
            weighted_preds = np.zeros_like(base_predictions[0])
            for i, preds in enumerate(base_predictions):
                weighted_preds += preds * norm_weights[i]
                
            # Calculer la MAE
            return mean_absolute_error(y_val, weighted_preds)
        
        # Optimisation par recherche aléatoire
        best_mae = float('inf')
        best_weights = self.weights.copy()
        
        for _ in range(n_trials):
            # Générer des poids aléatoires
            random_weights = np.random.rand(len(self.models))
            # Normaliser
            random_weights = random_weights / random_weights.sum()
            
            # Calculer la MAE
            mae = calculate_mae(random_weights)
            
            # Mettre à jour si meilleur
            if mae < best_mae:
                best_mae = mae
                best_weights = random_weights.tolist()
                
        logger.info(f"Meilleurs poids trouvés: {best_weights}")
        logger.info(f"MAE avec poids optimisés: {best_mae:.4f}")
        
        # Mettre à jour les poids de l'ensemble
        self.weights = best_weights
        
        return best_weights
    
    def save_ensemble(self, directory='models/ensemble'):
        """
        Sauvegarde l'ensemble (poids et références aux modèles).
        
        Args:
            directory (str): Répertoire où sauvegarder l'ensemble
        """
        # Créer le répertoire si nécessaire
        os.makedirs(directory, exist_ok=True)
        
        # Sauvegarde des poids
        weights_path = os.path.join(directory, 'ensemble_weights.json')
        with open(weights_path, 'w') as f:
            json.dump(self.weights, f, indent=4)
        logger.info(f"Poids de l'ensemble sauvegardés à {weights_path}")
        
        # Sauvegarde des informations sur les modèles
        model_info = []
        for i, model in enumerate(self.models):
            if isinstance(model, XGBoostModel):
                model_type = 'XGBoost'
                model_path = os.path.join(directory, f'xgb_model_{i}.json')
                model.save_model(model_path)
            elif isinstance(model, LightGBMModel):
                model_type = 'LightGBM'
                model_path = os.path.join(directory, f'lgb_model_{i}.txt')
                model.save_model(model_path)
            else:
                model_type = 'Unknown'
                model_path = None
                logger.warning(f"Type de modèle non reconnu: {type(model)}")
                
            model_info.append({
                'type': model_type,
                'path': os.path.basename(model_path) if model_path else None,
                'weight': self.weights[i]
            })
            
        # Sauvegarde des informations sur l'ensemble
        info_path = os.path.join(directory, 'ensemble_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        logger.info(f"Informations de l'ensemble sauvegardées à {info_path}")
        
    def load_ensemble(self, directory='models/ensemble'):
        """
        Charge un ensemble préalablement sauvegardé.
        
        Args:
            directory (str): Répertoire où l'ensemble a été sauvegardé
        """
        # Vérifier si le répertoire existe
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Le répertoire {directory} n'existe pas.")
            
        # Charger les informations de l'ensemble
        info_path = os.path.join(directory, 'ensemble_info.json')
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Le fichier {info_path} n'existe pas.")
            
        with open(info_path, 'r') as f:
            model_info = json.load(f)
            
        # Réinitialiser les modèles et les poids
        self.models = []
        self.weights = []
        
        # Charger chaque modèle
        for info in model_info:
            model_type = info['type']
            model_path = os.path.join(directory, info['path']) if info['path'] else None
            weight = info['weight']
            
            if model_type == 'XGBoost' and model_path:
                model = XGBoostModel()
                model.load_model(model_path)
                self.models.append(model)
                self.weights.append(weight)
            elif model_type == 'LightGBM' and model_path:
                model = LightGBMModel()
                model.load_model(model_path)
                self.models.append(model)
                self.weights.append(weight)
            else:
                logger.warning(f"Impossible de charger le modèle de type {model_type}")
                
        logger.info(f"Ensemble chargé avec {len(self.models)} modèles.")
        logger.info(f"Poids: {self.weights}")
        
        return self
    
    def create_submission(self, X_test, output_file='submission_ensemble.csv'):
        """
        Crée un fichier de soumission avec les prédictions de l'ensemble.
        
        Args:
            X_test: Features de test
            output_file (str): Nom du fichier de sortie
        """
        predictions = self.predict(X_test)
        
        submission = pd.DataFrame({'p0q0': predictions})
        submission.to_csv(output_file, index=True, index_label='')
        
        logger.info(f"Fichier de soumission '{output_file}' créé avec succès!")
        logger.info("Statistiques des prédictions:")
        logger.info(f"Min: {submission['p0q0'].min():.2f}, Max: {submission['p0q0'].max():.2f}, "
                  f"Moyenne: {submission['p0q0'].mean():.2f}")
        
        return submission
