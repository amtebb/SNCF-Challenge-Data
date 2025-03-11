import pandas as pd

class DataEncoder:
    def __init__(self,x_train,x_test,y_train):
        self.x_train=pd.read_csv(x_train)
        self.x_test=pd.read_csv(x_test)

        self.y_train=pd.read_csv(y_train)
        self.xy=self.x_train.copy()
        self.xy['p0q0']=self.y_train['p0q0']
        
        self.final_df_train=self.x_train.copy()
        self.final_df_test=self.x_test.copy()
        self.gare_correspondance=self.calculer_gare_correspondance()
        self.encodage_train=self.calculer_train_correspondance()
        self.encode_dates()
        self.final_df_train=self.final_df_train.drop(['gare','train'], axis=1)
        self.final_df_test=self.final_df_test.drop(['gare','train'], axis=1)
        
        


    def get_df_encoded(self):
        return self.final_df_train.iloc[:, 3:], self.final_df_test.iloc[:, 3:]
            
    def calculer_gare_correspondance(self):
        groupe=self.xy.groupby('gare')['p0q0']
        mean=groupe.mean()
        std=groupe.std()
        med=groupe.median()
        stats = pd.DataFrame({
            'gare': mean.index,
            'mean': mean.values,
            'std': std.values,
            'median': med.values
        })
        stats.set_index('gare', inplace=True)
        
        # Count occurrences of each station in both train and test sets
        gare_x_train = self.x_train['gare']
        gare_x_test = self.x_test['gare']
        combined_gares = pd.concat([gare_x_train, gare_x_test])
        gare_counts = combined_gares.value_counts()
        
        # Create mapping dictionary for stations and their occurrences
        gare_encoding = pd.DataFrame({'gare': gare_counts.index, 'occurrences': gare_counts.values})
        
        # Map the occurrences to both train and test datasets
        self.final_df_train['gare_occurrence'] = self.final_df_train['gare'].map(gare_encoding.set_index('gare')['occurrences'])
        self.final_df_test['gare_occurrence'] = self.final_df_test['gare'].map(gare_encoding.set_index('gare')['occurrences'])
        
        # Map the statistics (mean, std, median) to both train and test datasets
        self.final_df_train['gare_mean'] = self.final_df_train['gare'].map(stats['mean'])
        self.final_df_train['gare_std'] = self.final_df_train['gare'].map(stats['std'])
        self.final_df_train['gare_median'] = self.final_df_train['gare'].map(stats['median'])
        
        self.final_df_test['gare_mean'] = self.final_df_test['gare'].map(stats['mean'])
        self.final_df_test['gare_std'] = self.final_df_test['gare'].map(stats['std'])
        self.final_df_test['gare_median'] = self.final_df_test['gare'].map(stats['median'])
        return stats
    def calculer_train_correspondance(self):
        train_x = self.x_train['train']
        test_x = self.x_test['train']
        combined_trains = pd.concat([train_x, test_x])
        train_counts = combined_trains.value_counts()
        train_encoding = pd.DataFrame({'train': train_counts.index, 'occurrences': train_counts.values})
        
        
        self.final_df_train['train_occurrences'] = self.final_df_train['train'].map(train_encoding.set_index('train')['occurrences'])
        self.final_df_test['train_occurrences'] = self.final_df_test['train'].map(train_encoding.set_index('train')['occurrences'])
        
        return train_encoding
    def process_date(self, df):
        
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['day'] = df['date'].dt.dayofweek
        
        
        hour_weights = {
            
            7: 5, 8: 6, 9: 4,
            
            16: 4, 17: 5, 18: 6, 19: 4,
            
            12: 3, 13: 3,
        }
        
        df['hour_weight'] = df['hour'].map(lambda x: hour_weights.get(x, 1))
        
        df['day_weight'] = df['day'].map(lambda x:  1 if x > 5 else 3)
        
        df['tension_score'] = df['hour_weight'] * df['day_weight']
        
        df.drop(['date', 'hour', 'day', 'hour_weight', 'day_weight'], axis=1, inplace=True)
        return df
    
    def encode_dates(self):
        self.final_df_train = self.process_date(self.final_df_train)
        self.final_df_test = self.process_date(self.final_df_test)
        return self.final_df_train, self.final_df_test



        
