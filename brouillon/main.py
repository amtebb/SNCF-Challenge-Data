import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from preprocess2 import enhanced_encoding

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

# Diviser pour la validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_encoded, y_train_full, test_size=0.2, random_state=42
)

print("Création des matrices DMatrix pour XGBoost...")

# Créer les matrices DMatrix pour XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Paramètres XGBoost optimisés
params = {
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

# Liste d'évaluation
evallist = [(dtrain, 'train'), (dval, 'validation')]

print("Entraînement du modèle XGBoost...")

# Entraînement du modèle
model = xgb.train(
    params,
    dtrain,
    num_boost_round=350,
    evals=evallist,
    early_stopping_rounds=20,
    verbose_eval=10
)

# Évaluation sur l'ensemble de validation
preds_val = model.predict(dval)
mae_val = mean_absolute_error(y_val, preds_val)
print(f"\nMAE sur l'ensemble de validation: {mae_val:.4f}")

# Features importantes
importance = model.get_score(importance_type='gain')
importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print("\nFeatures les plus importantes:")
for feat, score in importance[:10]:
    print(f"{feat}: {score:.2f}")

print("\nPréparation des prédictions finales...")

# Prédictions sur l'ensemble de test
dtest = xgb.DMatrix(X_test_encoded)
preds_test = model.predict(dtest)

# Création du fichier de soumission
submission = pd.DataFrame({'p0q0': preds_test})
submission.to_csv('submission_enhanced.csv', index=True, index_label='')

print("Fichier de soumission 'submission_enhanced.csv' créé avec succès!")
print("Statistiques des prédictions:")
print(f"Min: {submission['p0q0'].min():.2f}, Max: {submission['p0q0'].max():.2f}, Moyenne: {submission['p0q0'].mean():.2f}")
