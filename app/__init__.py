from sklearn.base import BaseEstimator, TransformerMixin
from contextlib import asynccontextmanager
from fastapi import FastAPI, Body
from joblib import load
import pandas as pd

class TimeInformation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        time_data = pd.to_datetime(X['hora_ref'])
        b = [0, 4, 8, 12, 16, 20, 24]
        l = ['late night', 'early morning', 'morning', 'noon', 'evening', 'night']
        X['period_day'] = pd.cut(time_data.dt.hour, bins=b, labels=l, include_lowest=True)
        X['day_of_week'] =  pd.Categorical(time_data.dt.day_of_week)
        return X

class CabeceiraTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['prev_troca_cabeceira'] = X['prev_troca_cabeceira'].astype(bool)
        X['troca_cabeceira_hora_anterior'] = X['troca_cabeceira_hora_anterior'].astype(bool)
        return X


class CombineMetarTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_features=None, num_features=None):
        self.cat_features = cat_features
        self.num_features = num_features
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for feature in self.cat_features:
            X[feature] = X[f'{feature}_metar'].fillna(X[f'{feature}_metaf'])
            # X.drop(columns=[f'{feature}_metar', f'{feature}_metaf'], inplace=True)
        for feature in self.num_features:
            X[feature] = X[[f'{feature}_metar', f'{feature}_metaf']].mean(axis=1)
            # X.drop(columns=[f'{feature}_metar', f'{feature}_metaf'], inplace=True)
        return X

class RouteTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['route'] = X['origem'] + '-' + X['destino']
        return X

class WeatherTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['current_wx_origem'] = X['current_wx1_origem'].fillna('') + X['current_wx2_origem'].fillna('') + X['current_wx3_origem'].fillna('')
        X['current_wx_destino'] = X['current_wx1_destino'].fillna('') + X['current_wx2_destino'].fillna('') + X['current_wx3_destino'].fillna('')

        intensity_cat = ['VC', '-', '', '+']
        descriptor_cat = ['MI', 'PR', 'BC', 'DR', 'BL', 'SH', 'TS', 'FZ']
        descriptor_cat = [i+d for i in intensity_cat for d in descriptor_cat]
        precipitation_cat = ['DZ', 'RA', 'SN', 'SG', 'IC', 'PL', 'GR', 'GS', 'UP']
        precipitation_cat = [i+d for i in intensity_cat for d in precipitation_cat]
        obscuration_cat = ['BR', 'FG', 'FU', 'VA', 'DU', 'SA', 'HZ', 'PY']
        obscuration_cat = [i+d for i in intensity_cat for d in obscuration_cat]
        other_cat = ['PO', 'SQ', 'FC', 'SS', 'DS']
        other_cat = [i+d for i in intensity_cat for d in other_cat]
        weather_categories = descriptor_cat + precipitation_cat + obscuration_cat + other_cat

        X['current_wx_origem'] = pd.Categorical(X['current_wx_origem'], categories=weather_categories)
        X['current_wx_destino'] = pd.Categorical(X['current_wx_destino'], categories=weather_categories)
        
        # X.drop(columns=[f'current_wx{i}_{source}' for i in range(1,4) for source in ['origem', 'destino']], inplace=True)
        return X

pd.options.mode.chained_assignment = None

model = load('models/new_clf_th042_auc_7953.joblib')

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     models['clf'] = load('models/new_clf_th042_auc_7953.joblib')
#     yield
#     models['clf'] = None

app = FastAPI(title='API', 
                description='API for inference', 
                version='1.0',
                # lifespan=lifespan,
                )