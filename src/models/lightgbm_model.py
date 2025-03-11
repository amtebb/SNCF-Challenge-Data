import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
import optuna
import logging
import os
import json
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LightGBMModel:
    """
    Classe pour le modèle LightGBM utilisé pour prédire les retards de trains.
    """
    def __init__(self, params=None):
        """
        Initialise le modèle LightGBM avec des paramètres optionnels.
        
        Args:
            params (dict, optional): Paramètres du modèle LightGBM.
        """
        self.model = None
        self.default_params = {
            'objective': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 50,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1
        }
        self.params = params if params is not None else self.default_params
        
    def train(self, X_train, y_train, X_val=None, y_val=None, num_boost_round=650, early_stopping_rounds=20):
        """
        Entraîne le modèle LightGBM.
        
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
        logger.info("Création des datasets LightGBM...")
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [train_data, val_data]
            valid_names = ['train', 'validation']
        else:
            valid_sets = [train_data]
            valid_names = ['train']
            
        logger.info("Entraînement du modèle LightGBM...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=True),
                lgb.log_evaluation(10)
            ]
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
            
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Évalue le modèle sur un ensemble de données.
        
        Args:
            X: Features pour l'évaluation
            y: Cible réelle
            
        Returns:
            float: Erreur absolue moyenne (MAE)
        """
        predictions = self.predict(X)
        mae = mean_absolute_error(y, predictions)
        logger.info(f"MAE sur l'ensemble d'évaluation: {mae:.4f}")
        return mae
    
    def feature_importance(self, n_top=10, plot=False, figsize=(12, 6)):
        """
        Renvoie les caractéristiques les plus importantes du modèle.
        
        Args:
            n_top (int): Nombre de caractéristiques à afficher
            plot (bool): Si True, affiche un graphique des importances
            figsize (tuple): Taille du graphique si plot=True
            
        Returns:
            pandas.DataFrame: DataFrame des caractéristiques les plus importantes
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez d'abord la méthode train().")
            
        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.model.feature_name()
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        logger.info("Features les plus importantes:")
        for i, (feat, score) in enumerate(zip(importance_df['Feature'].head(n_top), 
                                             importance_df['Importance'].head(n_top))):
            logger.info(f"{feat}: {score}")
            
        if plot:
            plt.figure(figsize=figsize)
            plt.barh(
                importance_df['Feature'].head(n_top)[::-1],
                importance_df['Importance'].head(n_top)[::-1]
            )
            plt.title('LightGBM Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            # Sauvegarder le graphique
            os.makedirs('figures', exist_ok=True)
            plt.savefig('figures/lgb_feature_importance.png')
            plt.close()
            
        return importance_df.head(n_top)
    
    def optimize_hyperparams(self, X_train, y_train, n_trials=30):
        """
        Optimise les hyperparamètres du modèle en utilisant Optuna.
        
        Args:
            X_train: Features d'entraînement
            y_train: Cible d'entraînement
            n_trials (int): Nombre d'essais pour l'optimisation
            
        Returns:
            dict: Meilleurs paramètres trouvés
        """
        def objective(trial):
            params = {
                "objective": "mae",
                "boosting_type": "gbdt",
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "max_depth": trial.suggest_int("max_depth", 5, 15),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "verbose": -1
            }
            
            # Validation croisée pour éviter le surapprentissage
            cv_scores = []
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for train_idx, val_idx in kf.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Entraînement sur ce fold
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                
                # Évaluation sur ce fold
                preds = model.predict(X_fold_val)
                fold_mae = mean_absolute_error(y_fold_val, preds)
                cv_scores.append(fold_mae)
            
            # Retourner la moyenne des scores
            mean_mae = np.mean(cv_scores)
            logger.info(f"Trial params: {params}, Mean MAE: {mean_mae:.6f}")
            
            return mean_mae

        logger.info("Début de l'optimisation des hyperparamètres...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_params["objective"] = "mae"
        best_params["boosting_type"] = "gbdt"
        best_params["verbose"] = -1
        
        logger.info(f"Meilleurs paramètres trouvés: {best_params}")
        logger.info(f"Meilleur score MAE: {study.best_value:.6f}")
        
        # Tracer l'historique d'optimisation
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title('Optuna Optimization History')
        plt.tight_layout()
        
        # Sauvegarder le graphique
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/lgb_optuna_history.png')
        plt.close()
        
        # Mettre à jour les paramètres du modèle
        self.params = best_params
        
        return best_params
    
    def save_model(self, model_path='models/lgb_model.txt'):
        """
        Sauvegarde le modèle entraîné.
        
        Args:
            model_path (str): Chemin où sauvegarder le modèle
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez d'abord la méthode train().")
            
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        self.model.save_model(model_path)
        logger.info(f"Modèle sauvegardé avec succès à {model_path}")
        
        # Sauvegarder également les paramètres
        params_path = os.path.join(os.path.dirname(model_path), 'lgb_params.json')
        with open(params_path, 'w') as f:
            json.dump(self.params, f, indent=4)
        logger.info(f"Paramètres sauvegardés avec succès à {params_path}")
        
    def load_model(self, model_path='models/lgb_model.txt'):
        """
        Charge un modèle préalablement entraîné.
        
        Args:
            model_path (str): Chemin du modèle à charger
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le fichier modèle {model_path} n'existe pas.")
            
        self.model = lgb.Booster(model_file=model_path)
        logger.info(f"Modèle chargé avec succès depuis {model_path}")
        
        # Charger également les paramètres si disponibles
        params_path = os.path.join(os.path.dirname(model_path), 'lgb_params.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                self.params = json.load(f)
            logger.info(f"Paramètres chargés avec succès depuis {params_path}")
        
        return self
    
    def create_submission(self, X_test, output_file='submission_lgb.csv'):
        """
        Crée un fichier de soumission avec les prédictions du modèle.
        
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
