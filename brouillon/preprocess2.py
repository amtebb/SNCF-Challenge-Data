import pandas as pd
import numpy as np

def enhanced_encoding(X_train_full, X_test_full, y_train_full):
    # Créer une copie pour travailler avec les données
    X_train = X_train_full.copy()
    X_test = X_test_full.copy()
    
    # Combiner X_train et y_train pour calculer des statistiques
    train_with_target = X_train.copy()
    train_with_target['p0q0'] = y_train_full['p0q0']
    
    # Conversion des dates
    train_with_target['date'] = pd.to_datetime(train_with_target['date'])
    train_with_target['day_of_week'] = train_with_target['date'].dt.dayofweek
    train_with_target['hour'] = train_with_target['date'].dt.hour
    train_with_target['is_weekend'] = (train_with_target['day_of_week'] >= 5).astype(int)
    train_with_target['month'] = train_with_target['date'].dt.month
    
    X_train['date'] = pd.to_datetime(X_train['date'])
    X_train['day_of_week'] = X_train['date'].dt.dayofweek
    X_train['hour'] = X_train['date'].dt.hour
    X_train['is_weekend'] = (X_train['day_of_week'] >= 5).astype(int)
    X_train['month'] = X_train['date'].dt.month
    
    X_test['date'] = pd.to_datetime(X_test['date'])
    X_test['day_of_week'] = X_test['date'].dt.dayofweek
    X_test['hour'] = X_test['date'].dt.hour
    X_test['is_weekend'] = (X_test['day_of_week'] >= 5).astype(int)
    X_test['month'] = X_test['date'].dt.month
    
    # Score de tension selon le jour et l'heure
    def get_tension_score(row):
        day = row['day_of_week']
        hour = row['hour']
        
        # Heures de pointe
        peak_hours = {7: 5, 8: 7, 9: 4, 16: 4, 17: 5, 18: 7, 19: 4, 12: 3, 13: 3}
        hour_score = peak_hours.get(hour, 1)
        
        # Jours de semaine vs weekend
        day_score = 3 if day < 5 else 1
        
        return hour_score * day_score
    
    train_with_target['tension_score'] = train_with_target.apply(get_tension_score, axis=1)
    X_train['tension_score'] = X_train.apply(get_tension_score, axis=1)
    X_test['tension_score'] = X_test.apply(get_tension_score, axis=1)
    
    # 1. Statistiques par gare
    gare_stats = train_with_target.groupby('gare').agg({
        'p0q0': ['mean', 'std', 'median', 'min', 'max', 'count'],
        'tension_score': ['mean']
    })
    gare_stats.columns = ['_'.join(col).strip() for col in gare_stats.columns.values]
    gare_stats = gare_stats.reset_index()
    
    # 2. Statistiques par gare et jour de semaine
    gare_day_stats = train_with_target.groupby(['gare', 'day_of_week']).agg({
        'p0q0': ['mean', 'std', 'count']
    })
    gare_day_stats.columns = ['gare_day_' + '_'.join(col).strip() for col in gare_day_stats.columns.values]
    gare_day_stats = gare_day_stats.reset_index()
    
    # 3. Statistiques par gare pour heures de pointe vs heures creuses
    train_with_target['is_peak'] = train_with_target['tension_score'] > 3
    gare_peak_stats = train_with_target.groupby(['gare', 'is_peak']).agg({
        'p0q0': ['mean', 'std']
    })
    gare_peak_stats.columns = ['gare_peak_' + '_'.join(col).strip() for col in gare_peak_stats.columns.values]
    gare_peak_stats = gare_peak_stats.reset_index()
    
    # 4. Statistiques par gare et weekend/semaine
    gare_weekend_stats = train_with_target.groupby(['gare', 'is_weekend']).agg({
        'p0q0': ['mean', 'std']
    })
    gare_weekend_stats.columns = ['gare_weekend_' + '_'.join(col).strip() for col in gare_weekend_stats.columns.values]
    gare_weekend_stats = gare_weekend_stats.reset_index()
    
    # 5. Différence de retard entre stations consécutives
    gare_retard_diff = train_with_target.groupby(['train']).apply(
        lambda x: x.sort_values('arret')['p0q0'].diff()
    ).reset_index(name='diff_retard')
    
    # Calculer des statistiques sur ces différences par gare
    train_with_target = pd.merge(
        train_with_target, 
        gare_retard_diff, 
        on=['train'], 
        how='left'
    )
    
    gare_diff_stats = train_with_target.groupby('gare').agg({
        'diff_retard': ['mean', 'std', 'median']
    })
    gare_diff_stats.columns = ['gare_diff_' + '_'.join(col).strip() for col in gare_diff_stats.columns.values]
    gare_diff_stats = gare_diff_stats.reset_index()
    
    # 6. Fiabilité de la gare (variance des retards)
    train_with_target['abs_p0q0'] = train_with_target['p0q0'].abs()
    gare_reliability = train_with_target.groupby('gare').agg({
        'abs_p0q0': ['mean', 'std', 'median']
    })
    gare_reliability.columns = ['gare_reliability_' + '_'.join(col).strip() for col in gare_reliability.columns.values]
    gare_reliability = gare_reliability.reset_index()
    
    # Fusionner toutes les statistiques avec les données d'entraînement et de test
    X_train = pd.merge(X_train, gare_stats, on='gare', how='left')
    X_test = pd.merge(X_test, gare_stats, on='gare', how='left')
    
    # Fusionner les statistiques par jour de semaine
    X_train = pd.merge(X_train, gare_day_stats, on=['gare', 'day_of_week'], how='left')
    X_test = pd.merge(X_test, gare_day_stats, on=['gare', 'day_of_week'], how='left')
    
    # Fusionner les statistiques d'heures de pointe
    X_train['is_peak'] = X_train['tension_score'] > 3
    X_test['is_peak'] = X_test['tension_score'] > 3
    X_train = pd.merge(X_train, gare_peak_stats, on=['gare', 'is_peak'], how='left')
    X_test = pd.merge(X_test, gare_peak_stats, on=['gare', 'is_peak'], how='left')
    
    # Fusionner les statistiques de weekend
    X_train = pd.merge(X_train, gare_weekend_stats, on=['gare', 'is_weekend'], how='left')
    X_test = pd.merge(X_test, gare_weekend_stats, on=['gare', 'is_weekend'], how='left')
    
    # Fusionner les statistiques de différence de retard
    X_train = pd.merge(X_train, gare_diff_stats, on=['gare'], how='left')
    X_test = pd.merge(X_test, gare_diff_stats, on=['gare'], how='left')
    
    # Fusionner les statistiques de fiabilité
    X_train = pd.merge(X_train, gare_reliability, on=['gare'], how='left')
    X_test = pd.merge(X_test, gare_reliability, on=['gare'], how='left')
    
    # Créer des features d'interaction et d'agrégation
    for df in [X_train, X_test]:
        # Moyennes des délais passés
        df['moy_p0q'] = (df['p0q2'] + df['p0q3'] + df['p0q4']) / 3
        df['moy_pq0'] = (df['p2q0'] + df['p3q0'] + df['p4q0']) / 3
        
        # Retard cumulatif
        df['retard_cumulatif'] = df['p0q2'] + df['p0q3'] + df['p0q4']
        
        # Tendance du retard
        df['tendance_retard'] = df['p0q2'] - df['p0q4']
        
        # Retard relatif par rapport à la moyenne de la gare
        df['retard_relatif_gare'] = df['p0q2'] - df['p0q0_mean']
        
        # Impact du score de tension
        df['tension_impact'] = df['tension_score'] * df['p0q0_mean']
        
        # Interactions clés
        df['p0q2_p0q3'] = df['p0q2'] * df['p0q3']
        df['p0q3_p0q4'] = df['p0q3'] * df['p0q4']
        df['tension_p0q2'] = df['tension_score'] * df['p0q2']
    
    # Gérer les valeurs manquantes
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Supprimer les colonnes non nécessaires
    cols_to_drop = ['date', 'gare', 'train']
    X_train = X_train.drop(cols_to_drop, axis=1)
    X_test = X_test.drop(cols_to_drop, axis=1)
    
    return X_train, X_test