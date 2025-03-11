import pandas as pd
import numpy as np

class DataEncoder:
    """
    Classe pour encoder les données de retard des trains Transilien SNCF.
    Implémente l'encodage basique des caractéristiques.
    """
    def __init__(self, x_train_path, x_test_path, y_train_path):
        """
        Initialise l'encodeur avec les chemins des fichiers de données.
        
        Args:
            x_train_path: Chemin vers le fichier des features d'entraînement
            x_test_path: Chemin vers le fichier des features de test
            y_train_path: Chemin vers le fichier de la cible d'entraînement
        """
        self.x_train = pd.read_csv(x_train_path, index_col=0)
        self.x_test = pd.read_csv(x_test_path, index_col=0)
        self.y_train = pd.read_csv(y_train_path, index_col=0)
        
        # Fusion des données d'entraînement avec la cible pour les statistiques
        self.xy = self.x_train.copy()
        self.xy['p0q0'] = self.y_train['p0q0']
        
        # Initialisation des dataframes finaux
        self.final_df_train = self.x_train.copy()
        self.final_df_test = self.x_test.copy()
        
        # Suppression des colonnes inutiles (Unnamed)
        self.final_df_train = self.final_df_train.drop(columns=[col for col in self.final_df_train.columns if 'Unnamed' in col])
        self.final_df_test = self.final_df_test.drop(columns=[col for col in self.final_df_test.columns if 'Unnamed' in col])
        
        # Application des encodages
        self.gare_correspondance = self.calculer_gare_correspondance()
        self.encodage_train = self.calculer_train_correspondance()
        self.encode_dates()
        
        # Suppression des colonnes catégorielles originales
        self.final_df_train = self.final_df_train.drop(['gare', 'train'], axis=1)
        self.final_df_test = self.final_df_test.drop(['gare', 'train'], axis=1)

    def get_df_encoded(self):
        """
        Renvoie les dataframes encodés (train et test).
        
        Returns:
            tuple: (dataframe_train_encodé, dataframe_test_encodé)
        """
        return self.final_df_train.iloc[:, 3:], self.final_df_test.iloc[:, 3:]
            
    def calculer_gare_correspondance(self):
        """
        Calcule les statistiques par gare et les ajoute aux dataframes.
        
        Returns:
            pandas.DataFrame: Statistiques par gare
        """
        # Calcul des statistiques par gare
        groupe = self.xy.groupby('gare')['p0q0']
        mean = groupe.mean()
        std = groupe.std()
        med = groupe.median()
        
        stats = pd.DataFrame({
            'gare': mean.index,
            'mean': mean.values,
            'std': std.values,
            'median': med.values
        })
        stats.set_index('gare', inplace=True)
        
        # Comptage des occurrences de chaque gare
        gare_x_train = self.x_train['gare']
        gare_x_test = self.x_test['gare']
        combined_gares = pd.concat([gare_x_train, gare_x_test])
        gare_counts = combined_gares.value_counts()
        
        # Création du mapping pour les gares et leurs occurrences
        gare_encoding = pd.DataFrame({'gare': gare_counts.index, 'occurrences': gare_counts.values})
        
        # Application du mapping aux dataframes
        self.final_df_train['gare_occurrence'] = self.final_df_train['gare'].map(
            gare_encoding.set_index('gare')['occurrences'])
        self.final_df_test['gare_occurrence'] = self.final_df_test['gare'].map(
            gare_encoding.set_index('gare')['occurrences'])
        
        # Application des statistiques
        self.final_df_train['gare_mean'] = self.final_df_train['gare'].map(stats['mean'])
        self.final_df_train['gare_std'] = self.final_df_train['gare'].map(stats['std'])
        self.final_df_train['gare_median'] = self.final_df_train['gare'].map(stats['median'])
        
        self.final_df_test['gare_mean'] = self.final_df_test['gare'].map(stats['mean'])
        self.final_df_test['gare_std'] = self.final_df_test['gare'].map(stats['std'])
        self.final_df_test['gare_median'] = self.final_df_test['gare'].map(stats['median'])
        
        return stats
    
    def calculer_train_correspondance(self):
        """
        Calcule les statistiques par train et les ajoute aux dataframes.
        
        Returns:
            pandas.DataFrame: Encodage des trains
        """
        
        train_x = self.x_train['train']
        test_x = self.x_test['train']
        combined_trains = pd.concat([train_x, test_x])
        train_counts = combined_trains.value_counts()
        
        train_encoding = pd.DataFrame({'train': train_counts.index, 'occurrences': train_counts.values})
        
        
        self.final_df_train['train_occurrences'] = self.final_df_train['train'].map(
            train_encoding.set_index('train')['occurrences'])
        self.final_df_test['train_occurrences'] = self.final_df_test['train'].map(
            train_encoding.set_index('train')['occurrences'])
        
        return train_encoding
    
    def process_date(self, df):
        """
        Traite les dates pour en extraire des caractéristiques pertinentes.
        
        Args:
            df: DataFrame contenant une colonne 'date'
            
        Returns:
            pandas.DataFrame: DataFrame avec les nouvelles caractéristiques temporelles
        """
        # Conversion de la date
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['day'] = df['date'].dt.dayofweek
        
        # Poids pour les heures de pointe
        hour_weights = {
            7: 5, 8: 6, 9: 4,  # Matin
            16: 4, 17: 5, 18: 6, 19: 4,  # Soir
            12: 3, 13: 3,  # Midi
        }
        
        # Application des poids
        df['hour_weight'] = df['hour'].map(lambda x: hour_weights.get(x, 1))
        df['day_weight'] = df['day'].map(lambda x: 1 if x > 5 else 3)  # Weekend vs semaine
        
        # Score de tension (plus élevé = plus de risque de retard)
        df['tension_score'] = df['hour_weight'] * df['day_weight']
        
        # Suppression des colonnes temporaires
        df.drop(['date', 'hour', 'day', 'hour_weight', 'day_weight'], axis=1, inplace=True)
        return df
    
    def encode_dates(self):
        """
        Applique l'encodage des dates aux dataframes train et test.
        
        Returns:
            tuple: (final_df_train, final_df_test)
        """
        self.final_df_train = self.process_date(self.final_df_train)
        self.final_df_test = self.process_date(self.final_df_test)
        return self.final_df_train, self.final_df_test
