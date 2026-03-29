import pandas as pd
import numpy as np
import os

class DataManager:
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.base_path = base_path
        
    def _standardize_columns(self, df):
        """Standardizes column names using aggressive cleaning and lookup."""
        # Aggressive cleaning: lower, strip, remove non-alphanumeric except underscore
        raw_cols = df.columns.tolist()
        df.columns = [c.encode('ascii', 'ignore').decode('ascii').strip().lower() for c in df.columns]
        df.columns = df.columns.str.replace(r'[^a-z0-9_]', '', regex=True)
        
        mapping = {
            'n': ['nitrogen'], # nitrogen contains n, so we match it
            'p': ['phosphorous', 'phosphorus'],
            'k': ['potassium'],
            'ph': ['ph', 'soil_ph'],
            'rainfall': ['rainfall', 'moisture', 'rain', 'precip'],
            'temperature': ['temperature', 'temp'],
            'humidity': ['humidity', 'hum'],
            'label': ['label', 'item', 'target']
        }
        
        new_cols_map = {}
        for col in df.columns:
            # First try exact match
            matched = False
            for standard, variations in mapping.items():
                if col == standard:
                    new_cols_map[col] = standard
                    matched = True
                    break
            if matched: continue
            
            # Then try substring match
            for standard, variations in mapping.items():
                if any(v in col for v in variations):
                    new_cols_map[col] = standard
                    break
        
        df = df.rename(columns=new_cols_map)
        
        # De-duplicate: Keep first occurrence of standard columns
        cols = pd.Series(df.columns)
        for dupe in cols[cols.duplicated()].unique():
            # If duplicated name is one of our standards, we only keep the first one
            pass
        df = df.loc[:, ~df.columns.duplicated()]

        # Ensure all standard keys are present
        for std in ['n', 'p', 'k', 'ph', 'rainfall', 'temperature', 'humidity']:
            if std not in df.columns:
                df[std] = 0.0
                
        return df

    def load_yield_data(self):
        """Loads and prepares yield prediction data."""
        path = os.path.join(self.base_path, "crop yield prediction/yield_df.csv")
        df = pd.read_csv(path, encoding='utf-8-sig')
        df = self._standardize_columns(df)
        
        # Filter for tomatoes if possible
        if 'item' in df.columns:
            df_tomato = df[df['item'].str.contains('Tomato', case=False, na=False)]
            if not df_tomato.empty:
                df = df_tomato
        
        # Synthesize missing features: Plant height, Leaf count, Flower count
        num_samples = len(df)
        df['plant_height'] = np.random.uniform(30, 150, num_samples)
        df['leaf_count'] = np.random.randint(10, 100, num_samples)
        df['flower_count'] = np.random.randint(0, 30, num_samples)
        
        # Specifically handle the 'hg/ha_yield' column which might be messy
        potential_yield_cols = [c for c in df.columns if 'yield' in c or 'hgha' in c]
        if potential_yield_cols:
            df = df.rename(columns={potential_yield_cols[0]: 'yield_val'})
        
        if 'yield_val' not in df.columns:
             df['yield_val'] = 0.0

        return df

    def load_fertilizer_data(self):
        """Loads and prepares fertilizer recommendation data."""
        path = os.path.join(self.base_path, "fertilizer remmonder/fertilizer_recommendation.csv")
        df = pd.read_csv(path, encoding='utf-8-sig')
        # Explicit renaming before generic standardize to prevent wrong substring matches
        rename_map = {}
        for c in df.columns:
            if 'recommended_fertilizer' in c.lower() or 'fertilizer' == c.lower().strip():
                rename_map[c] = 'label'
        df = df.rename(columns=rename_map)
        df = self._standardize_columns(df)
        
        # Synthesize missing features if necessary
        if 'temperature' not in df.columns:
            df['temperature'] = np.random.uniform(15, 40, len(df))
        if 'humidity' not in df.columns:
            df['humidity'] = np.random.uniform(30, 90, len(df))
            
        return df

    def load_soil_data(self):
        """Loads and prepares soil type prediction data."""
        path = os.path.join(self.base_path, "soil texture/Crop_recommendation.csv")
        df = pd.read_csv(path, encoding='utf-8-sig')
        df = self._standardize_columns(df)
        
        # Synthesize soil texture if not present (mapping from NPK/pH/Rainfall)
        # Normal soil types: Clay, Loamy, Sandy
        conditions = [
            (df['rainfall'] > 150),
            (df['ph'] < 6.0),
            (df['ph'] >= 6.0) & (df['rainfall'] <= 150)
        ]
        choices = ['Clay', 'Sandy', 'Loamy']
        df['soil_type'] = np.select(conditions, choices, default='Loamy')
        
        # Moisture type
        df['moisture_type'] = np.where(df['rainfall'] > 100, 'Wet', 'Dry')
        
        return df

if __name__ == "__main__":
    dm = DataManager()
    print("Yield Data Head:\n", dm.load_yield_data().head())
    print("Fertilizer Data Head:\n", dm.load_fertilizer_data().head())
    print("Soil Data Head:\n", dm.load_soil_data().head())
