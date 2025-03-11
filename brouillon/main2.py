import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from preprocess2 import enhanced_encoding
# Supposons que X_train_encoded et X_test_encoded sont déjà préparés avec votre encodage amélioré
# Lors du chargement des données
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

# Diviser les données pour la validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_encoded, y_train_full, test_size=0.2, random_state=42
)

#train_data = lgb.Dataset(X_train, label=y_train)
#val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
import optuna
import lightgbm as lgb
from sklearn.model_selection import cross_val_score


def objective(trial):
    params = {
        "objective": "mae",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),  # Correction ici
        "num_leaves": trial.suggest_int("num_leaves", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10, log=True),  # Correction ici
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),  # Correction ici
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),  # Correction ici
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),  # Correction ici
        "n_estimators": 500,
        "verbose": -1
    }

    model = lgb.LGBMRegressor(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error").mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

best_params = study.best_params
print(best_params)




'''
params = {'bagging_fraction': 0.7367911977625198, 
 'bagging_freq': 6, 
 'feature_fraction': 0.9715741174592356, 
 'lambda_l1': 4.923403520671534,
   'lambda_l2': 7.938777074118048, 
   'learning_rate': 0.09514403528190511,
     'max_depth': 9, 'min_data_in_leaf': 40,
       'min_gain_to_split': 0.00028735474045558473,
         'num_leaves': 195, 'objective': 'mae', 
         'boosting_type': 'gbdt', 
         'verbose': -1}
'''
'''     
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
'''
'''
# Entraînement du modèle
print("Entraînement du modèle LightGBM...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=650,
    valid_sets=[train_data, val_data],
    callbacks=[lgb.early_stopping(20, verbose=True),
               lgb.log_evaluation(10)]
)

# Évaluation sur l'ensemble de validation
preds_val = model.predict(X_val)
mae_val = mean_absolute_error(y_val, preds_val)
print(f"\nMAE sur l'ensemble de validation: {mae_val:.4f}")

# Features importantes
importance = model.feature_importance(importance_type='gain')
feature_names = model.feature_name()
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
importance_df = importance_df.sort_values('Importance', ascending=False)
print("\nFeatures les plus importantes:")
for i, (feat, score) in enumerate(zip(importance_df['Feature'].head(10), importance_df['Importance'].head(10))):
    print(f"{feat}: {score}")

# Prédictions sur l'ensemble de test
print("\nPréparation des prédictions finales...")
preds_test = model.predict(X_test_encoded)

# Création du fichier de soumission
submission = pd.DataFrame({'p0q0': preds_test})
submission.to_csv('submission_lightgbm.csv', index=True, index_label='')

print("Fichier de soumission 'submission_lightgbm.csv' créé avec succès!")
print("Statistiques des prédictions:")
print(f"Min: {submission['p0q0'].min():.2f}, Max: {submission['p0q0'].max():.2f}, Moyenne: {submission['p0q0'].mean():.2f}")
'''