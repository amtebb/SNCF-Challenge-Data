import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
from preprocess2 import enhanced_encoding

# Supposons que X_train_encoded et y_train_full sont déjà préparés
X_train_full = pd.read_csv('x_train_final.csv', index_col=0)
X_test_full = pd.read_csv('x_test_final.csv', index_col=0)
y_train_full = pd.read_csv('y_train_final_j5KGWWK.csv', index_col=0)

# Ou après chargement
X_train_full = X_train_full.drop(columns=[col for col in X_train_full.columns if 'Unnamed' in col])
X_test_full = X_test_full.drop(columns=[col for col in X_test_full.columns if 'Unnamed' in col])
print("Données chargées. Application de l'encodage amélioré...")

# Appliquer l'encodage amélioré
X_train_encoded, X_test_encoded = enhanced_encoding(X_train_full, X_test_full, y_train_full)

print(f"Encodage terminé. Forme des données d'entraînement: {X_train_encoded.shape}")
print(f"Encodage terminé. Forme des données de test: {X_test_encoded.shape}")

# Diviser pour la validation finale
X_train, X_val, y_train, y_val = train_test_split(
    X_train_encoded, y_train_full, test_size=0.2, random_state=42
)

print("Optimisation des hyperparamètres de LightGBM...")

# Définir l'espace de recherche des hyperparamètres
space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.1)),
    'num_leaves': hp.quniform('num_leaves', 20, 200, 1),
    'max_depth': hp.quniform('max_depth', 5, 15, 1),
    'min_data_in_leaf': hp.quniform('min_data_in_leaf', 5, 50, 1),
    'feature_fraction': hp.uniform('feature_fraction', 0.7, 1.0),
    'bagging_fraction': hp.uniform('bagging_fraction', 0.7, 1.0),
    'bagging_freq': hp.quniform('bagging_freq', 1, 10, 1),
    'lambda_l1': hp.loguniform('lambda_l1', np.log(1e-8), np.log(10.0)),
    'lambda_l2': hp.loguniform('lambda_l2', np.log(1e-8), np.log(10.0)),
    'min_gain_to_split': hp.loguniform('min_gain_to_split', np.log(1e-8), np.log(1.0))
}

# Fonction d'objectif pour l'optimisation avec validation croisée
def objective(params):
    # Convertir certains paramètres en entiers
    params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth'])
    params['min_data_in_leaf'] = int(params['min_data_in_leaf'])
    params['bagging_freq'] = int(params['bagging_freq'])
    
    # Paramètres fixes
    fixed_params = {
        'objective': 'mae',
        'boosting_type': 'gbdt',
        'verbose': -1
    }
    
    # Combiner les paramètres
    params.update(fixed_params)
    
    # Validation croisée
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data]
        )
        
        preds = model.predict(X_fold_val)
        mae = mean_absolute_error(y_fold_val, preds)
        mae_scores.append(mae)
    
    avg_mae = np.mean(mae_scores)
    print(f"Params: {params}, MAE: {avg_mae:.6f}")
    
    return {'loss': avg_mae, 'status': STATUS_OK}

# Exécuter l'optimisation Bayésienne
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,  # Ajustez selon votre temps disponible
    trials=trials
)

print("\nMeilleurs hyperparamètres trouvés:")
for param, value in best.items():
    if param in ['num_leaves', 'max_depth', 'min_data_in_leaf', 'bagging_freq']:
        best[param] = int(value)
    print(f"{param}: {best[param]}")

# Tracer l'évolution des scores
plt.figure(figsize=(12, 6))
losses = [trial['result']['loss'] for trial in trials.trials]
best_till_now = np.minimum.accumulate(losses)
plt.plot(range(1, len(losses) + 1), losses, 'o-', label='MAE par essai')
plt.plot(range(1, len(losses) + 1), best_till_now, 'r--', label='Meilleur MAE')
plt.xlabel('Nombre d\'essais')
plt.ylabel('MAE')
plt.title('Progrès de l\'optimisation des hyperparamètres')
plt.legend()
plt.grid(True)
plt.savefig('hyperopt_progress.png')

# Entraîner le modèle final avec les meilleurs hyperparamètres
best_params = {
    'objective': 'mae',
    'boosting_type': 'gbdt',
    'verbose': -1
}
best_params.update(best)

print("\nEntraînement du modèle final avec les meilleurs hyperparamètres...")

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

final_model = lgb.train(
    best_params,
    train_data,
    num_boost_round=2000,  # Plus d'itérations pour le modèle final
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(50, verbose=True),
               lgb.log_evaluation(100)]
)

# Évaluer le modèle final
val_preds = final_model.predict(X_val)
final_mae = mean_absolute_error(y_val, val_preds)
print(f"\nMAE final sur validation: {final_mae:.6f}")

# Corriger les biais systématiques
val_residuals = y_val.values.flatten() - val_preds
X_val_with_gare = pd.DataFrame({
    'gare': X_train_full.loc[X_val.index, 'gare'].values,
    'residual': val_residuals
})

gare_bias = X_val_with_gare.groupby('gare')['residual'].mean().reset_index()
gare_bias.columns = ['gare', 'bias']
'''
# Prédictions sur le test
test_preds = final_model.predict(X_test_encoded)

# Appliquer la correction de biais
X_test_with_gare = pd.DataFrame({
    'gare': X_test_full['gare'].values
})
X_test_with_gare = pd.merge(X_test_with_gare, gare_bias, on='gare', how='left')
X_test_with_gare['bias'] = X_test_with_gare['bias'].fillna(0)

final_predictions = test_preds + X_test_with_gare['bias'].values

# Création du fichier de soumission
submission = pd.DataFrame({'p0q0': final_predictions})
submission.to_csv('submission_optimal_lgb.csv', index=True, index_label='')

print("Fichier de soumission 'submission_optimal_lgb.csv' créé avec succès!")
print("Statistiques des prédictions:")
print(f"Min: {submission['p0q0'].min():.2f}, Max: {submission['p0q0'].max():.2f}, Moyenne: {submission['p0q0'].mean():.2f}")
'''