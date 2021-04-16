import tsfresh
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from tsfresh.transformers.feature_augmenter import FeatureAugmenter

from python_connection.datastructure.datastructure import PandasDataFrameWrapper

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin



class TsFreshBasedFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, default_fc_parameters=None,
                 kind_to_fc_parameters=None, column_id="instance_id", column_sort="timestep",
                 column_kind="sensor", column_value="value",
                 chunksize=tsfresh.defaults.CHUNKSIZE,
                 n_jobs=0, show_warnings=tsfresh.defaults.SHOW_WARNINGS,
                 disable_progressbar=True,
                 impute_function=tsfresh.defaults.IMPUTE_FUNCTION,
                 profile=tsfresh.defaults.PROFILING,
                 profiling_filename=tsfresh.defaults.PROFILING_FILENAME,
                 profiling_sorting=tsfresh.defaults.PROFILING_SORTING
                 ):
        self.default_fc_parameters = default_fc_parameters
        self.kind_to_fc_parameters = kind_to_fc_parameters

        self.column_id = column_id
        self.column_sort = column_sort
        self.column_kind = column_kind
        self.column_value = column_value

        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.show_warnings = show_warnings
        self.disable_progressbar = disable_progressbar
        self.impute_function = impute_function
        self.profile = profile
        self.profiling_filename = profiling_filename
        self.profiling_sorting = profiling_sorting


        self.feature_extractor = self.create_tsfresh_feature_augmenter()

    def fit(self, X, y):
        return self

    def transform(self, X):
        #assume X is arff_parsed
        structured_data = PandasDataFrameWrapper(X)

        #create empty container for extracted features
        extracted_features = pd.DataFrame(index=structured_data.y.index)
        timeseries_container = structured_data.X

        self.feature_extractor.set_timeseries_container(timeseries_container)

        extracted_features = self.feature_extractor.transform(extracted_features)

        invalid_number_replacer = InvalidNumberReplacementTransformer()
        return invalid_number_replacer.transform(extracted_features)


    def create_tsfresh_feature_augmenter(self):
        return FeatureAugmenter(default_fc_parameters=self.default_fc_parameters,
                                kind_to_fc_parameters=self.kind_to_fc_parameters,
                                column_id=self.column_id,
                                column_sort=self.column_sort,
                                column_kind=self.column_kind,
                                column_value=self.column_value,
                                n_jobs=self.n_jobs,
                                chunksize=self.chunksize,
                                show_warnings=self.show_warnings,
                                disable_progressbar=self.disable_progressbar,
                                impute_function=self.impute_function,
                                profile=self.profile,
                                profiling_filename=self.profiling_filename,
                                profiling_sorting=self.profiling_sorting
                                )


class LowComputationTimeFCParameters(dict):
    def __init__(self):
        initial_map = ComprehensiveFCParameters()
        initial_map.pop("sample_entropy")
        initial_map.pop("change_quantiles")
        initial_map.pop("approximate_entropy")
        initial_map.pop("number_cwt_peaks")
        initial_map.pop("augmented_dickey_fuller")
        initial_map.pop("quantile")
        initial_map.pop("agg_linear_trend")
        initial_map.pop("max_langevin_fixed_point")
        initial_map.pop("friedrich_coefficients")
        initial_map.pop("fft_coefficient")
        initial_map.pop("large_standard_deviation")
        initial_map.pop("autocorrelation")
        initial_map.pop("cwt_coefficients")
        initial_map.pop("percentage_of_reoccurring_values_to_all_values")
        initial_map.pop("ar_coefficient")
        initial_map.pop("ratio_beyond_r_sigma")
        initial_map.pop("number_peaks")
        initial_map.pop("linear_trend_timewise") #broken
        super().__init__(initial_map)

class MidComputationTimeFCParameters(dict):
    def __init__(self):
        initial_map = ComprehensiveFCParameters()
        initial_map.pop("sample_entropy")
        initial_map.pop("change_quantiles")
        initial_map.pop("linear_trend_timewise")  # broken
        super().__init__(initial_map)

class FCParametersDictionary(dict):
    def __init__(self, has_duplicate_max, binned_entropy, last_location_of_maximum, abs_energy, c3, value_count, mean_second_derivative_central, first_location_of_minimum, standard_deviation, length, mean_abs_change, has_duplicate_min, mean_change, sum_values, percentage_of_reoccurring_datapoints_to_all_datapoints, range_count, absolute_sum_of_changes, energy_ratio_by_chunks, last_location_of_minimum, linear_trend, variance_larger_than_standard_deviation, spkt_welch_density, cid_ce, symmetry_looking, has_duplicate, skewness, count_above, count_above_mean, longest_strike_below_mean, mean, agg_autocorrelation, ratio_value_number_to_time_series_length, fft_aggregated, first_location_of_maximum, partial_autocorrelation, sum_of_reoccurring_data_points, count_below, count_below_mean, variance, longest_strike_above_mean, median, kurtosis, minimum, time_reversal_asymmetry_statistic, number_crossing_m, sum_of_reoccurring_values, maximum, approximate_entropy, number_cwt_peaks, augmented_dickey_fuller, quantile, agg_linear_trend, max_langevin_fixed_point, friedrich_coefficients, fft_coefficient, large_standard_deviation, autocorrelation, cwt_coefficients, percentage_of_reoccurring_values_to_all_values, ar_coefficient, ratio_beyond_r_sigma, number_peaks, sample_entropy, change_quantiles, index_mass_quantile):
        initial_map = ComprehensiveFCParameters()
        initial_map.pop("linear_trend_timewise")  # broken
        if not has_duplicate_max:
            initial_map.pop("has_duplicate_max")
        if not binned_entropy:
            initial_map.pop("binned_entropy")
        if not last_location_of_maximum:
            initial_map.pop("last_location_of_maximum")
        if not abs_energy:
            initial_map.pop("abs_energy")
        if not c3:
            initial_map.pop("c3")
        if not value_count:
            initial_map.pop("value_count")
        if not mean_second_derivative_central:
            initial_map.pop("mean_second_derivative_central")
        if not first_location_of_minimum:
            initial_map.pop("first_location_of_minimum")
        if not standard_deviation:
            initial_map.pop("standard_deviation")
        if not length:
            initial_map.pop("length")
        if not mean_abs_change:
            initial_map.pop("mean_abs_change")
        if not has_duplicate_min:
            initial_map.pop("has_duplicate_min")
        if not mean_change:
            initial_map.pop("mean_change")
        if not sum_values:
            initial_map.pop("sum_values")
        if not percentage_of_reoccurring_datapoints_to_all_datapoints:
            initial_map.pop("percentage_of_reoccurring_datapoints_to_all_datapoints")
        if not range_count:
            initial_map.pop("range_count")
        if not absolute_sum_of_changes:
            initial_map.pop("absolute_sum_of_changes")
        if not energy_ratio_by_chunks:
            initial_map.pop("energy_ratio_by_chunks")
        if not last_location_of_minimum:
            initial_map.pop("last_location_of_minimum")
        if not linear_trend:
            initial_map.pop("linear_trend")
        if not variance_larger_than_standard_deviation:
            initial_map.pop("variance_larger_than_standard_deviation")
        if not spkt_welch_density:
            initial_map.pop("spkt_welch_density")
        if not cid_ce:
            initial_map.pop("cid_ce")
        if not symmetry_looking:
            initial_map.pop("symmetry_looking")
        if not has_duplicate:
            initial_map.pop("has_duplicate")
        if not skewness:
            initial_map.pop("skewness")
        if not count_above:
            initial_map.pop("count_above")
        if not count_above_mean:
            initial_map.pop("count_above_mean")
        if not longest_strike_below_mean:
            initial_map.pop("longest_strike_below_mean")
        if not mean:
            initial_map.pop("mean")
        if not agg_autocorrelation:
            initial_map.pop("agg_autocorrelation")
        if not ratio_value_number_to_time_series_length:
            initial_map.pop("ratio_value_number_to_time_series_length")
        if not fft_aggregated:
            initial_map.pop("fft_aggregated")
        if not first_location_of_maximum:
            initial_map.pop("first_location_of_maximum")
        if not partial_autocorrelation:
            initial_map.pop("partial_autocorrelation")
        if not sum_of_reoccurring_data_points:
            initial_map.pop("sum_of_reoccurring_data_points")
        if not count_below:
            initial_map.pop("count_below")
        if not count_below_mean:
            initial_map.pop("count_below_mean")
        if not variance:
            initial_map.pop("variance")
        if not longest_strike_above_mean:
            initial_map.pop("longest_strike_above_mean")
        if not median:
            initial_map.pop("median")
        if not kurtosis:
            initial_map.pop("kurtosis")
        if not minimum:
            initial_map.pop("minimum")
        if not time_reversal_asymmetry_statistic:
            initial_map.pop("time_reversal_asymmetry_statistic")
        if not number_crossing_m:
            initial_map.pop("number_crossing_m")
        if not sum_of_reoccurring_values:
            initial_map.pop("sum_of_reoccurring_values")
        if not maximum:
            initial_map.pop("maximum")
        if not approximate_entropy:
            initial_map.pop("approximate_entropy")
        if not number_cwt_peaks:
            initial_map.pop("number_cwt_peaks")
        if not augmented_dickey_fuller:
            initial_map.pop("augmented_dickey_fuller")
        if not quantile:
            initial_map.pop("quantile")
        if not agg_linear_trend:
            initial_map.pop("agg_linear_trend")
        if not max_langevin_fixed_point:
            initial_map.pop("max_langevin_fixed_point")
        if not friedrich_coefficients:
            initial_map.pop("friedrich_coefficients")
        if not fft_coefficient:
            initial_map.pop("fft_coefficient")
        if not large_standard_deviation:
            initial_map.pop("large_standard_deviation")
        if not autocorrelation:
            initial_map.pop("autocorrelation")
        if not cwt_coefficients:
            initial_map.pop("cwt_coefficients")
        if not percentage_of_reoccurring_values_to_all_values:
            initial_map.pop("percentage_of_reoccurring_values_to_all_values")
        if not ar_coefficient:
            initial_map.pop("ar_coefficient")
        if not ratio_beyond_r_sigma:
            initial_map.pop("ratio_beyond_r_sigma")
        if not number_peaks:
            initial_map.pop("number_peaks")
        if not sample_entropy:
            initial_map.pop("sample_entropy")
        if not change_quantiles:
            initial_map.pop("change_quantiles")
        if not index_mass_quantile:
            initial_map.pop("index_mass_quantile")
        super().__init__(initial_map)


class InvalidNumberReplacementTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        extracted_features = np.float32(X.to_numpy())
        features = np.nan_to_num(extracted_features)
        return features