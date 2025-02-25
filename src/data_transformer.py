"""Module containing functions for data transformations and calculations

REMARK: Functions defined here depend on configuration parameters in src/config.py
"""
# Standard library imports

# Third-party imports
import numpy as np
import pandas as pd
# from scipy.spatial.transform import Rotation as R
# from scipy.signal import welch, hilbert, spectrogram, cwt, morlet2
# from numpy.fft import fft, ifft
# from scipy.fft import rfft, rfftfreq
# from scipy.stats import skew, kurtosis
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import OneHotEncoder
# from stingray import lightcurve
# from stingray.bispectrum import Bispectrum
# from statsmodels.tsa.stattools import pacf
# from pyrqa.time_series import TimeSeries
# from pyrqa.settings import Settings
# from pyrqa.analysis_type import Classic
# from pyrqa.neighbourhood import FixedRadius
# from pyrqa.metric import EuclideanMetric
# from pyrqa.computation import RQAComputation
# from pandarallel import pandarallel
# import swifter  # TODO: Fix swifter, it stopped working after installing fastparquet
# import nolds

# Local imports
import config as cfg
from utils import get_repo_path


def df_chopper(df, category, scalar_columns):
    """Cuts long time series contained in df into chunks defined by labels

        Args:
            df - dataframe with signals as columns
            category - column to chop dataframe by
            scalar_columns - columns with scalars
            columns - columns to take, if 'all' takes all the columns

        Returns:
            Dataframe with signals cut into chunks
    """

    def create_row(df):
        """Converts df to one-row-dataframe with columns of df converted to numpy-arrays"""

        new_df = pd.DataFrame({column: [np.array(df[column])] for column in columns})

        scalars = df[df.columns[df.columns.isin(scalar_columns)]].iloc[0]  # filter empty scalars
        new_df[df.columns[df.columns.isin(scalar_columns)]] = scalars
        return new_df

    columns = df.columns.drop([scalar_columns], errors='ignore')

    # Detecting indices where labels change
    encoded_labels = df[category].astype('category').cat.codes.reset_index(drop=True)
    segments = encoded_labels[encoded_labels.diff() != 0].index.tolist()

    splitted = np.split(df, segments)[1:]  # we remove the first element as it is an empty df
    return pd.concat(map(create_row, splitted), ignore_index=True)


# TODO: Refactor windower-function
def df_windower(df, window_size, window_step, scalar_columns):
    """Cuts signals in df into windows of the given size

        Args:
            df - dataframe with signals as columns
            window_size - number of samples in the window
            window_step - step used to cut signal into windows
            scalar_columns - list of columns in df containing scalars (those columns are not cut into windows)
        Returns:
            New dataframe with signals cut into windows
    """

    def create_window(array, row_no):
        """Returns list of windows taken from array.
        If array is too short prints warning indicating row number (row_no)"""
        window_list = []
        for i in range(0, array.size - window_size, window_step):
            window = array[i: i + window_size]
            window_list.append(window)

        if not len(window_list):
            print(f'WARNING: Row number {row_no} is too short for the window size.')

        return window_list

    def rowwise_windower(row):
        """Returns dataframe with windows created from each column of a given row;
        with exception for column label_col, whose values are taken without modification."""

        # apply create_window column-wise, except scalar columns
        new_row = row[df.columns.drop(scalar_columns, errors='ignore')].apply(lambda x: create_window(x, row.name))

        # from new_row containing lists of windows we create return a dataframe
        # .transpose() needed since apply understands columns as case indices
        new_df = new_row.apply(lambda x: pd.Series(x, dtype='object')).transpose()

        scalars = row[df.columns[df.columns.isin(scalar_columns)]]  # filter empty scalars
        new_df[df.columns[df.columns.isin(scalar_columns)]] = scalars
        return new_df

    if window_step > window_size:
        print(f"WARNING: STEP({window_step}) > SIZE({window_size}): SOME DATA WILL BE SKIPPED!")

    new_df = pd.concat(df.apply(rowwise_windower, axis=1).to_list())

    # for event, count in data_lost.items():
    #     data_left.update({event: original_data[event]-count})

    # with Logger(get_repo_path() / cfg.Prepare.log_file):
    #     print("\n##############################################")
    #     print("DATASET CLASSES STATISTICS")
    #     print("\nOriginal data:")
    #     print(original_data)
    #     print("\nData lost from dataset:")
    #     print(data_lost)
    #     print("\nData left in dataset:")
    #     print(data_left)
    #     print("\nData after creating windows:")
    #     print(new_df[cfg.General.label_column].value_counts())
    #     print("##############################################\n")
    return new_df.reset_index(drop=True)


def df_signals_windower(df, window_size, window_step):
    """Cuts signals in df into windows of the given size

        Args:
            df - dataframe with signals as columns
            window_size - number of samples in the window
            window_step - step used to cut signal into windows
        Returns:
            New dataframe with signals cut into windows
    """

    def create_window(array):
        """Returns list of windows taken from array."""
        window_list = []

        for i in range(0, array.size - window_size, window_step):
            window = array[i: i + window_size]
            window_list.append(window)
        return window_list

    def rowwise_windower(row):
        """Returns dataframe with windows created from each column of a given row."""

        # apply create_window column-wise
        new_row = row.apply(create_window)

        # from new_row containing lists of windows we create return a dataframe
        # .transpose() needed since apply understands columns as case indices
        new_df = new_row.apply(lambda x: pd.Series(x, dtype='object')).transpose()
        return new_df

    if window_step > window_size:
        print(f"WARNING: STEP({window_step}) > SIZE({window_size}): SOME DATA WILL BE SKIPPED!")

    new_df = pd.concat(df.apply(rowwise_windower, axis=1).to_list())
    return new_df.reset_index(drop=True)

#
# @pd.api.extensions.register_dataframe_accessor('comp')
# class Compute:
#
#     def __init__(self, df):
#         self.df = df
#
#     def mean(self, column):
#         return self.df[column].apply(np.mean)
#
#     def stddev(self, column):
#         return self.df[column].apply(np.std)
#
#     def kurtosis(self, column):
#         return self.df[column].apply(kurtosis)
#
#     def skewness(self, column):
#         return self.df[column].apply(skew)
#
#     def fft(self, column):
#         """Returns FFT of the given column containing (REAL!) signals"""
#         return self.df[column].apply(rfft)
#
#     def fftfreq(self, column):
#         """Returns vectors of frequencies for the given column
#
#         Args:
#             column - column with time-domain signals (IMPORTANT!)
#
#         """
#         return self.df[column].apply(lambda x: rfftfreq(x.size, cfg.General.delta_t))
#
#     def abs(self, column):
#         return self.df[column].apply(np.abs)
#
#     def fft_features(self, column, n, fmin=0, fmax=float('inf'), phase=False):
#         """Returns pandas-dataframe with mean-value of abs(FFT) in equally-spaced frequency intervals"""
#
#         def split(row):
#             freq = rfftfreq(row.size, cfg.General.delta_t)
#             ffts = rfft(row)
#             condition = (freq >= fmin) & (freq <= fmax)
#             fft_restricted = ffts[condition]
#             result = np.array_split(fft_restricted, n)
#             return result
#
#         splitted = self.df[column].apply(split)
#
#         if phase:
#             return splitted.apply(lambda x: [np.mean(np.angle(array)) for array in x])
#         else:
#             return splitted.apply(lambda x: [np.mean(abs(array)) for array in x])
#
#     def log_fft_features(self, column, n, fmin=0, fmax=float('inf'), phase=False):
#         fft_features = self.fft_features(column, n, fmin, fmax, phase)
#         return fft_features.apply(lambda x: [np.log(value) for value in x])
#
#     def power_spectrum(self, column):
#         """Calculates power spectrum using Welch algorithm"""
#         return self.df[column].apply(lambda x: welch(x, fs=1 / cfg.General.delta_t, nperseg=x.size))
#
#     def hilbert(self, column):
#         """Hilbert transform (computes analytic signal)"""
#         return self.df[column].apply(hilbert)
#
#     def amplitude(self, column):
#         """Instantaneous amplitude"""
#         return self.hilbert(column).apply(np.abs)
#
#     def phase(self, column):
#         """Instantaneous phase"""
#         return self.hilbert(column).apply(np.angle).apply(np.unwrap)
#
#     def frequency(self, column):
#         """Instantaneous frequency"""
#         return self.phase(column).apply(np.diff) / cfg.General.delta_t
#
#     def phase_std(self, column):
#         """Standard deviation of the non-linear component of instantaneous phase"""
#
#         def nonlinear_component_std(x):
#             least_squares = LinearRegression()
#             index = np.arange(x.size).reshape(-1, 1)
#             least_squares.fit(index, x)
#             linear_trend = least_squares.predict(index)
#
#             return np.std(x - linear_trend)
#
#         return self.df[column].apply(nonlinear_component_std)
#
#     def spectrogram(self, column, fs=1 / cfg.General.delta_t, **kwargs):
#         """Spectrogram of the given signal"""
#         return self.df[column].apply(lambda x: spectrogram(x, fs, **kwargs))
#
#     def wavelet(self, column, f_min, f_max, n=50, mode='psd'):
#         """Spectrogram of the given signal"""
#         w0 = (5 + np.sqrt(27)) / 2
#         s_min = w0 / (2 * np.pi * cfg.General.delta_t * f_max)
#         s_max = w0 / (2 * np.pi * cfg.General.delta_t * f_min)
#         widths = np.linspace(start=s_min, stop=s_max, num=n)
#
#         if mode == 'psd':
#             return self.df[column].apply(lambda x: np.power(np.abs(cwt(x,
#                                                                        morlet2,
#                                                                        widths=widths,
#                                                                        dtype=np.complex128)), 2))
#         elif mode == 'abs':
#             return self.df[column].apply(lambda x: np.abs(cwt(x,
#                                                               morlet2,
#                                                               widths=widths,
#                                                               dtype=np.complex128)))
#         elif mode == 'phase':
#             return self.df[column].apply(lambda x: np.angle(cwt(x,
#                                                                 morlet2,
#                                                                 widths=widths,
#                                                                 dtype=np.complex128)))
#
#     def higher_order_moment(self, column, p, q):
#
#         def calculate_moment(signal, p, q):
#             return np.mean(signal ** (p - q) * np.conj(signal) ** q)
#
#         signals = self.hilbert(column)
#         return signals.apply(lambda x: calculate_moment(x, p, q))
#
#     def cumulant(self, column, p, q):
#
#         if p == 2 and q == 0:
#             M20 = self.higher_order_moment(column, 2, 0)
#             return abs(M20)
#         elif p == 2 and q == 1:
#             M11 = self.higher_order_moment(column, 1, 1)
#             return abs(M11)
#         elif p == 4 and q == 0:
#             M40 = self.higher_order_moment(column, 4, 0)
#             M20 = self.higher_order_moment(column, 2, 0)
#             return abs(M40 - 3 * M20 ** 2)
#         elif p == 4 and q == 1:
#             M41 = self.higher_order_moment(column, 4, 1)
#             M20 = self.higher_order_moment(column, 2, 0)
#             M21 = self.higher_order_moment(column, 2, 0)
#             return abs(M41 - 3 * M21 * M20)
#         elif p == 4 and q == 2:
#             M42 = self.higher_order_moment(column, 4, 2)
#             M20 = self.higher_order_moment(column, 2, 0)
#             M21 = self.higher_order_moment(column, 2, 0)
#             return abs(M42 - abs(M20) ** 2 - 2 * M21 ** 2)
#         else:
#             raise NotImplementedError("Cumulant not implemented")
#
#     def log_cumulant(self, column, p, q):
#         cumulant = self.cumulant(column, p, q)
#         return np.log(cumulant)
#
#     def correlation(self, column, column2=None, normalise=False, separate_time=True):
#         """Calculates cross- or auto-correlation function
#
#         Args:
#         column - first column to cross- or auto- correlate
#         column2 - second column to cross-correlate. If None the function calculates auto-correlation
#         normalise - if True, the function calculates Pearson statistic
#
#         Returns:
#         cross- or auto-correlation
#
#         Remarks:
#         The correlation function as calculated by np.correlate is z[n]=∑x[k]y[k-n+l], assuming parameter mode='same'
#         and the value of l depends on the lengths of signals M = len(x) and N = len(y)
#         It is l= ⌊N/2⌋ if M = N.
#               l = N-1  if M > N.
#               l = N-M  if M < N.
#         The range of n is from 0 to max(M, N)-1
#         """
#
#         def correlate(x, y):
#             """Calculates (normalised) correlation function together with the time vector"""
#
#             M = len(x)
#             N = len(y)
#             if M == N:
#                 l = np.floor(N / 2)
#             elif M > N:
#                 l = N - 1
#             elif M < N:
#                 l = N - M
#
#             corr_function = np.correlate(x, y, mode='same')  # mode can be also valid or full
#             time_vec = (np.arange(corr_function.shape[0]) - l) * cfg.General.delta_t
#
#             if normalise:
#                 sum_sq_x = np.correlate(x ** 2, np.ones_like(y), mode='same')
#                 sum_x = np.correlate(x, np.ones_like(y), mode='same')
#                 sum_sq_y = np.correlate(np.ones_like(x), y ** 2, mode='same')
#                 sum_y = np.correlate(np.ones_like(x), y, mode='same')
#                 lengths = np.correlate(np.ones_like(x), np.ones_like(y))
#
#                 numerator = lengths * corr_function - sum_x * sum_y
#                 denominator = (np.sqrt((lengths * sum_sq_x - sum_x ** 2)) *
#                                np.sqrt((lengths * sum_sq_y - sum_y ** 2)))
#
#                 corr_function = numerator / denominator
#
#             return corr_function, time_vec
#
#         column2 = column2 or column
#
#         if separate_time:
#             corr_functions = self.df.apply(lambda x: correlate(x[column], x[column2]), axis=1)
#             return (corr_functions.apply(lambda x: x[0]),  # correlation functions
#                     corr_functions.apply(lambda x: x[1]))  # time vectors
#         else:
#             return self.df.apply(lambda x: correlate(x[column], x[column2]), axis=1)
#
#     def periodic_correlation(self, column):
#         """Calculates periodic auto-correlation function
#
#             Args:
#             column - first column to cross- or auto- correlate
#             column2 - second column to cross-correlate. If None the function calculates auto-correlation
#
#             Returns:
#             cross- or auto-correlation
#         """
#
#         def periodic_corr(x):
#             """Periodic correlation, implemented using the FFT."""
#             return ifft(fft(x) * fft(x).conj()).real
#
#         return self.df.apply(lambda x: periodic_corr(x[column]), axis=1)
#
#     def partial_autocorrelation(self, column, confidence=None):
#         """Calculates partial auto-correlation function
#
#             Args:
#             column - column to calculate auto-correlation function
#             confidence - If a number is given, the confidence intervals for the given level are returned.
#                          For instance if confidence=.05, 95 % confidence intervals are returned where
#                          the standard deviation is computed according to 1/sqrt(len(x)).
#
#             Returns:
#             partial auto-correlation
#             confidence intervals if confidence is not None.
#         """
#         return self.df[column].apply(lambda x: pacf(x, alpha=confidence, method='ywm'))
#
#     def correlation_features(self, n, column, column2=None, tmin=float('-inf'), tmax=float('inf')):
#         """Returns pandas-dataframe with correlation features in equally-spaced frequency intervals"""
#
#         def split(corr_values, corr_time):
#             if column2 is None:
#                 start_index = int(len(corr_time) / 2)
#                 condition = (corr_time >= tmin) & (corr_time <= tmax)
#                 corr_values_restricted = corr_values[condition]
#                 result = np.array_split(corr_values_restricted[start_index:], n)
#
#                 return result
#             else:
#                 condition = (corr_time >= tmin) & (corr_time <= tmax)
#                 corr_values_restricted = corr_values[condition]
#                 result = np.array_split(corr_values_restricted, n)
#
#                 return result
#
#         splitted = self.correlation(column, column2, separate_time=False).apply(lambda x: split(x[0], x[1]))
#         return splitted.apply(lambda x: [np.mean(abs(array)) for array in x])
#
#     def recurrence_features(self, column):
#         """Calculates recurrence features"""
#
#         def error_handler_for_recurrence_features(func):
#             """Decorator for handling errors. Returns empty pd.Series on error"""
#
#             def inner_function(*args, **kwargs):
#                 try:
#                     return func(*args, **kwargs)
#                 except Exception as e:
#                     return pd.Series()
#
#             return inner_function
#
#         @error_handler_for_recurrence_features
#         def calculate_recurrence_features(signal):
#             time_series = TimeSeries(signal,
#                                      embedding_dimension=1,
#                                      time_delay=2)
#             settings = Settings(time_series,
#                                 analysis_type=Classic,
#                                 neighbourhood=FixedRadius(10),
#                                 similarity_measure=EuclideanMetric,
#                                 theiler_corrector=1)
#             computation = RQAComputation.create(settings,
#                                                 verbose=False)
#             result = computation.run()
#
#             results = {'RR': result.recurrence_rate,  # Recurrence rate
#                        'DET': result.determinism,  # Determinism
#                        'L': result.average_diagonal_line,  # Average diagonal line length
#                        'L_max': result.longest_diagonal_line,  # Longest diagonal line length
#                        'DIV': result.divergence,  # Divergence
#                        'L_entr': result.entropy_diagonal_lines,  # Entropy diagonal lines
#                        'LAM': result.laminarity,  # Laminarity
#                        'TT': result.trapping_time,  # Trapping time
#                        'V_max': result.longest_white_vertical_line,  # Longest vertical line length
#                        'V_entr': result.entropy_vertical_lines,  # Entropy vertical lines
#                        'W': result.average_white_vertical_line,  # Average white vertical line length
#                        'W_max': result.longest_white_vertical_line,  # Longest white vertical line length
#                        'W_div': result.longest_white_vertical_line_inverse,  # Longest vertical line length inverse
#                        'W_entr': result.entropy_white_vertical_lines,  # Entropy white vertical lines
#                        'DET_RR': result.determinism / result.recurrence_rate,  # Ratio determinism / recurrence rate
#                        'LAM_DET': result.laminarity / result.determinism  # Ratio laminarity / determinism
#                        }
#
#             return pd.Series(results)
#
#         return self.df[column].swifter.apply(calculate_recurrence_features)
#
#     def bispectrum(self, column, window=None, scale='biased', maxlag=None):
#         def make_bispecs(array):
#             time = np.array(range(array.size)) * cfg.General.delta_t
#             signal = array
#             lc = lightcurve.Lightcurve(time, signal)
#
#             return Bispectrum(lc, window=window, scale=scale, maxlag=maxlag)
#
#         return self.df[column].apply(make_bispecs)
#
#     def correlation_dimension(self, column, emb_dim=2):
#         return self.df[column].swifter.apply(lambda x: nolds.corr_dim(x, emb_dim))
#
#
# class Augment:
#
#     @staticmethod
#     def mock_augmenter(row):
#         return row
#
#     @staticmethod
#     def gaussian_noise(row, sigma=0.05):
#         def add_noise(array):
#             noise = np.random.normal(0, sigma, array.size)
#             return np.add(array, noise)
#
#         return row.apply(add_noise)
#
#     @staticmethod
#     def random_rotations(row, sigma=5):
#         def rotate(vector, rotation, index):
#             rotated = np.array(rotation.apply(vector))
#             return pd.Series(rotated, index=index)
#
#         angle_x = np.random.normal(0, sigma)
#         angle_y = np.random.normal(0, sigma)
#         angle_z = np.random.normal(0, sigma)
#
#         r_x = R.from_euler('x', angle_x, degrees=True)
#         r_y = R.from_euler('y', angle_y, degrees=True)
#         r_z = R.from_euler('z', angle_z, degrees=True)
#
#         for vector_columns in cfg.General.signal_columns:
#             signal_rotated = pd.DataFrame(columns=vector_columns)
#
#             signal_vector = pd.DataFrame(row[vector_columns].to_list()).T
#
#             signal_rotated[vector_columns] = signal_vector.apply(lambda x: rotate(x, r_x, vector_columns), axis=1)
#             signal_rotated[vector_columns] = signal_rotated.apply(lambda x: rotate(x, r_y, vector_columns), axis=1)
#             signal_rotated[vector_columns] = signal_rotated.apply(lambda x: rotate(x, r_z, vector_columns), axis=1)
#
#             row[vector_columns[0]] = signal_rotated[vector_columns[0]].values
#             row[vector_columns[1]] = signal_rotated[vector_columns[1]].values
#             row[vector_columns[2]] = signal_rotated[vector_columns[2]].values
#         return row
#
#
# class WindowEncoder:
#
#     def __init__(self):
#         self._encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
#
#     def fit(self, y):
#         y = y.flatten().reshape(-1, 1)
#         self._encoder.fit(y)
#         return self
#
#     def transform(self, y):
#         def _encode(x):
#             zero_by_zero_matrix = np.array(x).reshape(-1, 1)
#             encoded_label = self._encoder.transform(zero_by_zero_matrix)
#             # return encoded_label.toarray()  # toarray needed because encoder returns sparse matrix
#             return encoded_label
#
#         tensor3d = np.apply_along_axis(_encode, 1, y)
#         window_size = tensor3d.shape[1]
#         return np.sum(tensor3d / window_size, axis=1)
#
#     def fit_transform(self, y):
#         self.fit(y)
#         return self.transform(y)
#
#     def inverse_transform(self, y):
#         return self._encoder.inverse_transform(y)
