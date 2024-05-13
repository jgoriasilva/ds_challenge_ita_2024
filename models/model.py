from lib2to3.pytree import Base
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from metpy.io import parse_metar
from metpy.io._metar_parser.metar_parser import ParseError

pd.options.mode.chained_assignment = None
PREDICT_THRESHOLD = 0.25

class ProcessMetarMetaf(BaseEstimator, TransformerMixin):
    def _parse_metar_metaf(self, X, metar_metaf: str):
        metarf = X.loc[:, ['hora_ref', metar_metaf]].drop_duplicates()
        parsed_metarfs = []
        for sample in metarf.index:
            data = metarf.loc[sample, metar_metaf]
            if data is None or isinstance(data, float):
                continue
            data = data.replace(metar_metaf.upper(), 'METAR')
            try:
                res = parse_metar(data, month=metarf.loc[sample, 'hora_ref'].month, year=metarf.loc[sample, 'hora_ref'].year)
            except ParseError:
                pass
            parsed_metarfs.append(res)
        metarf = pd.DataFrame(parsed_metarfs)
        metarf = metarf.rename(columns={'skyc1': 'low_cloud_type', 'skylev1': 'low_cloud_level',
                        'skyc2': 'medium_cloud_type', 'skylev2': 'medium_cloud_level',
                        'skyc3': 'high_cloud_type', 'skylev3': 'high_cloud_level',
                        'skyc4': 'highest_cloud_type', 'skylev4': 'highest_cloud_level',
                        'cloudcover': 'cloud_coverage', 'temperature': 'air_temperature',
                        'dewpoint': 'dew_point_temperature'})
        metarf = metarf.dropna(subset=['station_id', 'date_time'])
        metarf = metarf.drop_duplicates(subset=['station_id', 'date_time'])
        metarf['date_time'] = pd.to_datetime(metarf['date_time'], utc=True)
        return metarf
    
    def _merge_df_metarf(self, df1, df2, origem_destino, metar_metaf):
        return pd.merge(left=df1, right=df2.add_suffix(f'_{origem_destino}_{metar_metaf}'), left_on=['hora_ref', origem_destino], right_on=[f'date_time_{origem_destino}_{metar_metaf}',f'station_id_{origem_destino}_{metar_metaf}'], how='left')

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X['hora_ref'] = pd.to_datetime(X['hora_ref'], utc=True)
        metar = self._parse_metar_metaf(X, 'metar')
        metaf = self._parse_metar_metaf(X, 'metaf')
        X = self._merge_df_metarf(X, metar, 'origem', 'metar')
        X = self._merge_df_metarf(X, metar, 'destino', 'metar')
        X = self._merge_df_metarf(X, metaf, 'origem', 'metaf')
        X = self._merge_df_metarf(X, metaf, 'destino', 'metaf')
        return X


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

