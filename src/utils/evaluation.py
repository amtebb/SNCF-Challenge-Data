import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred, model_name="Modèle"):
    """
    Évalue un modèle en calculant différentes métriques.
    
    Args:
        y_true: Valeurs réelles
        y_pred: Prédictions du modèle
        model_name: Nom du modèle pour les logs
        
    Returns:
        dict: Dictionnaire des métriques calculées
    """
    # Calcul des métriques
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Calcul des erreurs relatives
    abs_error = np.abs(y_true - y_pred)
    rel_error = abs_error / (np.abs(y_true) + 1e-8)  # Éviter division par zéro
    mean_rel_error = np.mean(rel_error)
    
    # Affichage des résultats
    logger.info(f"Évaluation du {model_name}:")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"Erreur relative moyenne: {mean_rel_error:.4f}")
    
    # Retour des métriques dans un dictionnaire
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mean_rel_error': mean_rel_error
    }
    
    return metrics

def plot_prediction_analysis(y_true, y_pred, model_name="Modèle", save_dir="figures"):
    """
    Crée plusieurs visualisations pour analyser les prédictions d'un modèle.
    
    Args:
        y_true: Valeurs réelles
        y_pred: Prédictions du modèle
        model_name: Nom du modèle pour les titres
        save_dir: Répertoire où sauvegarder les figures
    """
    # Créer le répertoire si nécessaire
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Graphique de dispersion des prédictions vs valeurs réelles
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Prédictions')
    plt.title(f'{model_name} - Prédictions vs Valeurs réelles')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_scatter.png'))
    plt.close()
    
    # 2. Histogramme des erreurs
    errors = y_pred - y_true
    plt.figure(figsize=(10, 8))
    plt.hist(errors, bins=50, alpha=0.75)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Erreur de prédiction')
    plt.ylabel('Fréquence')
    plt.title(f'{model_name} - Distribution des erreurs')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_errors_hist.png'))
    plt.close()
    
    # 3. Graphique d'erreur absolue par rapport aux valeurs réelles
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, np.abs(errors), alpha=0.5)
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Erreur absolue')
    plt.title(f'{model_name} - Erreur absolue vs Valeurs réelles')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_abs_error.png'))
    plt.close()
    
    # 4. Boxplot des erreurs
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=errors)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.ylabel('Erreur de prédiction')
    plt.title(f'{model_name} - Boxplot des erreurs')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_errors_boxplot.png'))
    plt.close()
    
    logger.info(f"Graphiques d'analyse sauvegardés dans le répertoire {save_dir}")

def compare_models(results, metric='mae', save_dir="figures"):
    """
    Compare plusieurs modèles sur une métrique donnée.
    
    Args:
        results: Dictionnaire de la forme {nom_modèle: métriques}
        metric: Métrique à comparer
        save_dir: Répertoire où sauvegarder la figure
    """
    # Créer le répertoire si nécessaire
    os.makedirs(save_dir, exist_ok=True)
    
    # Extraire les valeurs et les noms
    models = list(results.keys())
    values = [results[model][metric] for model in models]
    
    # Création du graphique
    plt.figure(figsize=(12, 6))
    
    # Utiliser des couleurs différentes en fonction de la performance (plus c'est bas, mieux c'est pour MAE)
    colors = ['green' if metric in ['mae', 'mse', 'rmse', 'mean_rel_error'] else 'blue' for _ in values]
    
    # Trier les résultats pour une meilleure visualisation
    if metric in ['mae', 'mse', 'rmse', 'mean_rel_error']:
        # Pour ces métriques, plus c'est bas, mieux c'est
        sorted_indices = np.argsort(values)
    else:
        # Pour le R², plus c'est haut, mieux c'est
        sorted_indices = np.argsort(values)[::-1]
        
    sorted_models = [models[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    plt.bar(sorted_models, sorted_values, color=sorted_colors)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Modèles')
    
    # Afficher la bonne unité pour la métrique
    metric_labels = {
        'mae': 'MAE (minutes)',
        'mse': 'MSE (minutes²)',
        'rmse': 'RMSE (minutes)',
        'r2': 'R²',
        'mean_rel_error': 'Erreur relative moyenne'
    }
    plt.ylabel(metric_labels.get(metric, metric))
    
    plt.title(f'Comparaison des modèles - {metric_labels.get(metric, metric)}')
    plt.grid(axis='y')
    plt.tight_layout()
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(sorted_values):
        plt.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.savefig(os.path.join(save_dir, f'model_comparison_{metric}.png'))
    plt.close()
    
    logger.info(f"Graphique de comparaison pour {metric} sauvegardé dans {save_dir}")

def visualize_predictions_distribution(y_true, y_pred, model_name="Modèle", save_dir="figures"):
    """
    Visualise la distribution des valeurs réelles vs prédites.
    
    Args:
        y_true: Valeurs réelles
        y_pred: Prédictions du modèle
        model_name: Nom du modèle pour les titres
        save_dir: Répertoire où sauvegarder la figure
    """
    # Créer le répertoire si nécessaire
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Création de deux distributions
    sns.kdeplot(y_true, label='Valeurs réelles', fill=True, alpha=0.5)
    sns.kdeplot(y_pred, label='Prédictions', fill=True, alpha=0.5)
    
    plt.xlabel('Valeur de retard (minutes)')
    plt.ylabel('Densité')
    plt.title(f'{model_name} - Distribution des valeurs réelles vs prédites')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_distribution.png'))
    plt.close()
    
    logger.info(f"Graphique de distribution sauvegardé dans {save_dir}")

def analyze_errors_by_feature(X, y_true, y_pred, feature_name, model_name="Modèle", save_dir="figures", bins=10):
    """
    Analyse la relation entre les erreurs du modèle et une feature particulière.
    
    Args:
        X: DataFrame des features
        y_true: Valeurs réelles
        y_pred: Prédictions du modèle
        feature_name: Nom de la feature à analyser
        model_name: Nom du modèle pour les titres
        save_dir: Répertoire où sauvegarder la figure
        bins: Nombre de bins pour discrétiser la feature si elle est continue
    """
    # Créer le répertoire si nécessaire
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculer les erreurs
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    
    # Créer un DataFrame pour l'analyse
    analysis_df = pd.DataFrame({
        'feature': X[feature_name],
        'error': errors,
        'abs_error': abs_errors
    })
    
    # Si la feature est continue, la discrétiser en bins
    if analysis_df['feature'].dtype in [np.float64, np.float32, np.int64, np.int32]:
        analysis_df['feature_bin'] = pd.cut(analysis_df['feature'], bins=bins)
        feature_groups = analysis_df.groupby('feature_bin')
    else:
        feature_groups = analysis_df.groupby('feature')
    
    # Calculer l'erreur moyenne par groupe
    error_by_feature = feature_groups['abs_error'].mean().reset_index()
    
    # Créer le graphique
    plt.figure(figsize=(12, 6))
    
    if 'feature_bin' in error_by_feature.columns:
        # Pour les features discrétisées, utiliser un bar plot
        plt.bar(range(len(error_by_feature)), error_by_feature['abs_error'])
        plt.xticks(range(len(error_by_feature)), 
                  [str(x) for x in error_by_feature['feature_bin']], 
                  rotation=45, ha='right')
    else:
        # Pour les features catégorielles, utiliser un bar plot
        plt.bar(error_by_feature['feature'], error_by_feature['abs_error'])
        plt.xticks(rotation=45, ha='right')
    
    plt.xlabel(feature_name)
    plt.ylabel('Erreur absolue moyenne')
    plt.title(f'{model_name} - Erreur moyenne par {feature_name}')
    plt.grid(axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, f'{model_name.lower().replace(" ", "_")}_{feature_name}_error.png'))
    plt.close()
    
    logger.info(f"Graphique d'erreur par {feature_name} sauvegardé dans {save_dir}")
