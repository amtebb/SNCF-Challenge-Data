#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Importer les modules du projet
from src.preprocessing.enhanced_encoding import enhanced_encoding
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.ensemble import EnsembleModel
from src.utils.evaluation import (
    evaluate_model, 
    plot_prediction_analysis, 
    compare_models,
    visualize_predictions_distribution,
    analyze_errors_by_feature
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(data_dir="data"):
    """
    Charge les données d'entraînement et de test.
    
    Args:
        data_dir (str): Répertoire contenant les données
        
    Returns:
        tuple: (X_train_full, X_test_full, y_train_full)
    """
    logger.info("Chargement des données...")
    
    # Chemins des fichiers
    x_train_path = os.path.join(data_dir, "x_train_final.csv")
    x_test_path = os.path.join(data_dir, "x_test_final.csv")
    y_train_path = os.path.join(data_dir, "y_train_final_j5KGWWK.csv")
    
    # Vérifier que les fichiers existent
    for path in [x_train_path, x_test_path, y_train_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le fichier {path} n'existe pas.")
    
    # Chargement des données
    X_train_full = pd.read_csv(x_train_path, index_col=0)
    X_test_full = pd.read_csv(x_test_path, index_col=0)
    y_train_full = pd.read_csv(y_train_path, index_col=0)
    
    # Supprimer les colonnes inutiles
    X_train_full = X_train_full.drop(columns=[col for col in X_train_full.columns if 'Unnamed' in col], errors='ignore')
    X_test_full = X_test_full.drop(columns=[col for col in X_test_full.columns if 'Unnamed' in col], errors='ignore')
    
    logger.info(f"Données chargées. Forme des données d'entraînement: {X_train_full.shape}")
    logger.info(f"Forme des données de test: {X_test_full.shape}")
    logger.info(f"Forme de la cible d'entraînement: {y_train_full.shape}")
    
    return X_train_full, X_test_full, y_train_full

def train_xgboost(X_train, y_train, X_val, y_val, params=None, optimize=False):
    """
    Entraîne un modèle XGBoost.
    
    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        X_val: Features de validation
        y_val: Cible de validation
        params: Paramètres du modèle
        optimize: Si True, optimise les hyperparamètres
        
    Returns:
        XGBoostModel: Modèle entraîné
    """
    # Initialiser le modèle
    model = XGBoostModel(params)
    
    # Entraîner le modèle
    model.train(X_train, y_train, X_val, y_val)
    
    # Évaluer sur l'ensemble de validation
    val_mae = model.evaluate(X_val, y_val)
    
    # Afficher les caractéristiques importantes
    important_features = model.feature_importance(n_top=15)
    
    # Sauvegarder le modèle
    model.save_model()
    
    return model

def train_lightgbm(X_train, y_train, X_val, y_val, params=None, optimize=False):
    """
    Entraîne un modèle LightGBM.
    
    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        X_val: Features de validation
        y_val: Cible de validation
        params: Paramètres du modèle
        optimize: Si True, optimise les hyperparamètres
        
    Returns:
        LightGBMModel: Modèle entraîné
    """
    # Initialiser le modèle
    model = LightGBMModel(params)
    
    # Optimiser les hyperparamètres si demandé
    if optimize:
        logger.info("Optimisation des hyperparamètres LightGBM...")
        model.optimize_hyperparams(X_train, y_train, n_trials=30)
    
    # Entraîner le modèle
    model.train(X_train, y_train, X_val, y_val)
    
    # Évaluer sur l'ensemble de validation
    val_mae = model.evaluate(X_val, y_val)
    
    # Afficher les caractéristiques importantes
    important_features = model.feature_importance(n_top=15, plot=True)
    
    # Sauvegarder le modèle
    model.save_model()
    
    return model

def create_ensemble(X_train, y_train, X_val, y_val, models_to_include=None, optimize_weights=True):
    """
    Crée un ensemble de modèles.
    
    Args:
        X_train: Features d'entraînement
        y_train: Cible d'entraînement
        X_val: Features de validation
        y_val: Cible de validation
        models_to_include: Liste des types de modèles à inclure
        optimize_weights: Si True, optimise les poids de l'ensemble
        
    Returns:
        EnsembleModel: Modèle d'ensemble entraîné
    """
    ensemble = EnsembleModel()
    
    # Ajouter les modèles à l'ensemble
    if 'xgboost' in models_to_include:
        xgb_model = XGBoostModel()
        try:
            xgb_model.load_model()
            logger.info("Modèle XGBoost chargé.")
        except:
            logger.info("Entraînement d'un nouveau modèle XGBoost pour l'ensemble...")
            xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
        ensemble.add_model(xgb_model)
    
    if 'lightgbm' in models_to_include:
        lgb_model = LightGBMModel()
        try:
            lgb_model.load_model()
            logger.info("Modèle LightGBM chargé.")
        except:
            logger.info("Entraînement d'un nouveau modèle LightGBM pour l'ensemble...")
            lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
        ensemble.add_model(lgb_model)
    
    # Optimiser les poids si demandé
    if optimize_weights and len(ensemble.models) > 1:
        logger.info("Optimisation des poids de l'ensemble...")
        ensemble.optimize_weights(X_val, y_val)
    
    # Évaluer l'ensemble
    ensemble_mae = ensemble.evaluate(X_val, y_val)
    
    # Sauvegarder l'ensemble
    ensemble.save_ensemble()
    
    return ensemble

def main():
    """
    Fonction principale.
    """
    # Créer les répertoires nécessaires
    for directory in ['data', 'models', 'figures', 'submissions']:
        os.makedirs(directory, exist_ok=True)
    
    # Définir les arguments de ligne de commande
    parser = argparse.ArgumentParser(description='Prédiction des retards des trains Transilien SNCF')
    parser.add_argument('--model', type=str, default='lightgbm', choices=['xgboost', 'lightgbm', 'ensemble'],
                        help='Type de modèle à utiliser')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimiser les hyperparamètres')
    parser.add_argument('--ensemble_models', type=str, default='xgboost,lightgbm',
                        help='Liste des modèles à inclure dans l\'ensemble, séparés par des virgules')
    parser.add_argument('--predict', action='store_true',
                        help='Générer des prédictions sur l\'ensemble de test')
    parser.add_argument('--output', type=str, default='submissions/submission.csv',
                        help='Chemin du fichier de sortie pour les prédictions')
    parser.add_argument('--seed', type=int, default=42,
                        help='Graine aléatoire pour la reproductibilité')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Répertoire contenant les données')
    
    args = parser.parse_args()
    
    # Fixer la graine aléatoire
    np.random.seed(args.seed)
    
    # Charger les données
    X_train_full, X_test_full, y_train_full = load_data(args.data_dir)
    
    # Appliquer l'encodage amélioré
    logger.info("Application de l'encodage amélioré...")
    X_train_encoded, X_test_encoded = enhanced_encoding(X_train_full, X_test_full, y_train_full)
    logger.info(f"Encodage terminé. Forme des données d'entraînement: {X_train_encoded.shape}")
    logger.info(f"Forme des données de test: {X_test_encoded.shape}")
    
    # Diviser les données pour la validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_encoded, y_train_full, test_size=0.2, random_state=args.seed
    )
    
    # Sélectionner et entraîner le modèle
    model = None
    
    if args.model == 'xgboost':
        logger.info("Entraînement du modèle XGBoost...")
        model = train_xgboost(X_train, y_train, X_val, y_val, optimize=args.optimize)
        model_name = "XGBoost"
    
    elif args.model == 'lightgbm':
        logger.info("Entraînement du modèle LightGBM...")
        model = train_lightgbm(X_train, y_train, X_val, y_val, optimize=args.optimize)
        model_name = "LightGBM"
    
    elif args.model == 'ensemble':
        logger.info("Création de l'ensemble de modèles...")
        models_to_include = args.ensemble_models.split(',')
        model = create_ensemble(X_train, y_train, X_val, y_val, models_to_include, optimize_weights=args.optimize)
        model_name = "Ensemble"
    
    # Analyse des prédictions sur l'ensemble de validation
    if model is not None:
        logger.info("Analyse des prédictions sur l'ensemble de validation...")
        val_preds = model.predict(X_val)
        
        # Évaluer le modèle
        metrics = evaluate_model(y_val, val_preds, model_name)
        
        # Créer des visualisations
        plot_prediction_analysis(y_val, val_preds, model_name)
        visualize_predictions_distribution(y_val, val_preds, model_name)
        
        # Analyser les erreurs par rapport à certaines caractéristiques
        for feature in ['day_of_week', 'hour', 'is_weekend', 'tension_score']:
            if feature in X_val.columns:
                analyze_errors_by_feature(X_val, y_val, val_preds, feature, model_name)
    
    # Générer des prédictions sur l'ensemble de test si demandé
    if args.predict and model is not None:
        logger.info("Génération des prédictions sur l'ensemble de test...")
        output_file = args.output
        model.create_submission(X_test_encoded, output_file)
        logger.info(f"Prédictions sauvegardées dans {output_file}")

if __name__ == "__main__":
    main()
