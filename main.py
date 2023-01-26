import pandas as pd
import numpy as np
import logging
import warnings
from collections import OrderedDict
from fbprophet.models import StanBackendEnum


class MYProphet:
    def __init__(
            self,
            growth='linear',
            changepoints=None,
            n_changepoints=25,
            changepoint_range=0.8,
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality='auto',
            holidays=None,
            seasonality_mode='additive',
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            changepoint_prior_scale=0.05,
            mcmc_samples=0,
            interval_width=0.80,
            uncertainty_samples=1000,
            stan_backend=None
    ):
        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.specified_changepoints = False

        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays

        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = float(seasonality_prior_scale)
        self.changepoint_prior_scale = float(changepoint_prior_scale)
        self.holidays_prior_scale = float(holidays_prior_scale)

        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples

        # Set during fitting or by other methods
        self.start = None
        self.y_scale = None
        self.logistic_floor = False
        self.t_scale = None
        self.changepoints_t = None
        self.seasonalities = OrderedDict({})
        self.extra_regressors = OrderedDict({})
        self.country_holidays = None
        self.stan_fit = None
        self.params = {}
        self.history = None
        self.history_dates = None
        self.train_component_cols = None
        self.component_modes = None
        self.train_holiday_names = None
        self.fit_kwargs = {}
        self._load_stan_backend(stan_backend)

    def _load_stan_backend(self, stan_backend):
        if stan_backend is None:
            for i in StanBackendEnum:
                try:
                    return self._load_stan_backend(i.name)
                except Exception as e:
                    print("Unable to load backend %s (%s), trying the next one", i.name, e)
        else:
            self.stan_backend = StanBackendEnum.get_backend_class(stan_backend)()
        print("Loaded stan backend: %s", self.stan_backend.get_type())

    def initialize_scales(self, initialize_scales, df):
        pass

    def setup_dataframe(self, df, initialize_scales=False):
        if 'y' in df:
            df['y'] = pd.to_numeric(df['y'])
        if df['ds'].dtype == np.int64:
            df['ds'] = df['ds'].astype(str)
        df['ds'] = pd.to_datetime(df['ds'])
        for name in self.extra_regressors:
            df[name] = pd.to_numeric(df[name])
        for props in self.seasonalities.values():
            condition_name = props['condition_name']
            if condition_name is not None:
                df[condition_name] = df[condition_name].astype('bool')

        if df.index.name == 'ds':
            df.index.name = None
        df = df.sort_values('ds')
        df = df.reset_index(drop=True)




    def fit(self, df_, **kwargs):
        history = df[df['y'].notnull()].copy()
        self.history_dates = pd.to_datetime(pd.Series(df['ds'].unique(), name='ds')).sort_values()





if __name__ == '__main__':
    logging.getLogger('prophet').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')

    df = pd.read_csv("example_wp_log_peyton_manning.csv")

    playoffs = pd.DataFrame({
        'holiday': 'playoff',
        'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                              '2010-01-24', '2010-02-07', '2011-01-08',
                              '2013-01-12', '2014-01-12', '2014-01-19',
                              '2014-02-02', '2015-01-11', '2016-01-17',
                              '2016-01-24', '2016-02-07']),
        'lower_window': 0,
        'upper_window': 1,
    })
    superbowls = pd.DataFrame({
        'holiday': 'superbowl',
        'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
        'lower_window': 0,
        'upper_window': 1,
    })
    holidays = pd.concat((playoffs, superbowls))

    # df['ds'] = pd.to_datetime(df['ds'])
    # print(df['ds'].max() - df['ds'].min())

    model = MYProphet(holidays)
    model.fit(df)
    # future = model.make_future_dataframe(periods=366)