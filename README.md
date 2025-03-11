# Prédiction de retards des trains Transilien SNCF

Ce projet vise à prédire la différence entre le temps d'attente théorique et le temps d'attente réel des trains Transilien SNCF. Il a été développé dans le cadre d'un challenge compétitif.

## Contexte

Transilien SNCF Voyageurs est l'opérateur des trains de banlieue en Île-de-France, faisant circuler plus de 6 200 trains quotidiennement pour 3,4 millions de voyageurs. L'objectif est d'améliorer la qualité des prévisions de temps d'attente.

Le but spécifique est de prédire, à court terme, le temps d'attente d'un train situé deux gares en amont. La performance est évaluée en prédisant la différence entre le temps d'attente théorique et réel à chaque gare sur plusieurs jours.

## Structure du projet

```
├── data/                      # Dossier des données (à télécharger sur la page du challenge)
│   ├── x_train_final.csv      
│   ├── x_test_final.csv       
│   └── y_train_final_j5KGWWK.csv  
├── models/                    # Dossier pour sauvegarder les modèles
│   ├── xgb_model.json         
│   └── lgb_model.txt          
├── src/                       # Code source
│   ├── preprocessing/         
│   │   ├── __init__.py        
│   │   ├── basic_encoding.py  
│   │   └── enhanced_encoding.py  
│   ├── models/                # Scripts des modèles
│   │   ├── __init__.py        
│   │   ├── xgboost_model.py   
│   │   ├── lightgbm_model.py  
│   │   └── ensemble.py        
│   ├── utils/                 # Utilitaires
│   │   ├── __init__.py        
│   │   └── evaluation.py      
│   └── __init__.py            
├── main.py                    # Script principal d'exécution
├── requirements.txt           # Dépendances du projet
└── README.md                  # Ce fichier
```

## Installation et utilisation

### Prérequis

- Python 3.8+
- pip (gestionnaire de paquets Python)


## Approche méthodologique

### 1. Exploration et compréhension des données

- Analyse des distributions des variables
- Visualisation des corrélations
- Identification des caractéristiques temporelles importantes

### 2. Prétraitement des données


- Création de caractéristiques temporelles (heure, jour de la semaine, etc.)
- Encodage des variables catégorielles (gare, train)
- Création de caractéristiques avancées basées sur les statistiques par gare

### 3. Modélisation

L'approche a évolué en essayant différents algorithmes:

1. **XGBoost**: Premier modèle testé avec des paramètres de base
2. **Amélioration de l'encodage**: Développement d'un encodage plus sophistiqué
3. **LightGBM**: Test d'une alternative à XGBoost
4. **Optimisation des hyperparamètres**: Utilisation d'Optuna pour trouver les meilleurs paramètres
5. **Combinaison de modèles**: Tentative d'ensemble de modèles

### 4. Résultats

Le modèle LightGBM avec optimisation des hyperparamètres a donné les meilleurs résultats avec une MAE (Mean Absolute Error) de 0.66 sur l'ensemble de validation.

Les caractéristiques les plus importantes ont été:
- Les retards aux arrêts précédents du même train
- Le score de tension basé sur l'heure et le jour
- Les statistiques spécifiques à chaque gare

## Fichiers clés

- `src/preprocessing/enhanced_encoding.py`: Contient la logique de prétraitement améliorée
- `src/models/lightgbm_model.py`: Implémentation du modèle LightGBM avec optimisation
- `main.py`: Script principal pour entraîner, optimiser et générer des prédictions

