package autocoevorul.featurerextraction;

import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.BinaryIntegerVariable;
import org.moeaframework.core.variable.BinaryVariable;
import org.moeaframework.core.variable.RealVariable;

public class SolutionUtil {

	public static Solution getEmptySolution(final FeatureExtractionMoeaProblem problem, final GenomeHandler genomeHandler) {
		Solution solution = problem.newSolution();
		// ((BinaryVariable) solution.getVariable(genomeHandler.getIndexOfComponent("FCParametersDictionary"))).set(0, false);
		for (Integer index : genomeHandler.getIndexOfParameters("FCParametersDictionary")) {
			((BinaryIntegerVariable) solution.getVariable(index)).setValue(getPositionInArray("False", "True", "False"));
		}

		((BinaryVariable) solution.getVariable(genomeHandler.getIndexOfComponent("UltraFastShapeletsFeatureExtractor"))).set(0, false);
		((BinaryVariable) solution.getVariable(genomeHandler.getIndexOfComponent("ROCKET"))).set(0, false);

		// as RealVariables are initialized with NaN, we should set this to some value here to avoid problems even if it is not used
		int index = genomeHandler.getIndexOfParameter("UltraFastShapeletsFeatureExtractor", "keep_candidates_percentage");
		((RealVariable) solution.getVariable(index)).setValue(0.1);

		return solution;
	}

	public static Solution activateTsFresh(final Solution solution, final GenomeHandler genomeHandler) {
		// ((BinaryVariable) solution.getVariable(genomeHandler.getIndexOfComponent("FCParametersDictionary"))).set(0, true);

		String[] parameterName = new String[] { "has_duplicate_max", "binned_entropy", "last_location_of_maximum", "abs_energy", "c3", "value_count", "mean_second_derivative_central",
				"first_location_of_minimum", "standard_deviation", "length", "mean_abs_change", "has_duplicate_min", "mean_change", "sum_values",
				"percentage_of_reoccurring_datapoints_to_all_datapoints", "range_count", "absolute_sum_of_changes", "energy_ratio_by_chunks", "last_location_of_minimum", "linear_trend",
				"variance_larger_than_standard_deviation", "spkt_welch_density", "cid_ce", "symmetry_looking", "has_duplicate", "skewness", "count_above", "count_above_mean",
				"longest_strike_below_mean", "mean", "agg_autocorrelation", "ratio_value_number_to_time_series_length", "fft_aggregated", "first_location_of_maximum", "partial_autocorrelation",
				"sum_of_reoccurring_data_points", "count_below", "count_below_mean", "variance", "longest_strike_above_mean", "median", "kurtosis", "minimum", "time_reversal_asymmetry_statistic",
				"number_crossing_m", "sum_of_reoccurring_values", "maximum" };
		for (int i = 0; i < parameterName.length; i++) {
			int index = genomeHandler.getIndexOfParameter("FCParametersDictionary", parameterName[i]);
			((BinaryIntegerVariable) solution.getVariable(index)).setValue(getPositionInArray("True", "True", "False"));
		}
		activateRandomValueMaskingStrategy(solution, genomeHandler);
		return solution;
	}

	public static Solution activateRandomValueMaskingStrategy(final Solution solution, final GenomeHandler genomeHandler) {
		((BinaryVariable) solution.getVariable(genomeHandler.getIndexOfComponent("RandomValueMaskingStrategy"))).set(0, true);
		return solution;
	}

	public static Solution activateUltraFastShapelet(final Solution solution, final GenomeHandler genomeHandler) {
		((BinaryVariable) solution.getVariable(genomeHandler.getIndexOfComponent("UltraFastShapeletsFeatureExtractor"))).set(0, true);

		int index = genomeHandler.getIndexOfParameter("UltraFastShapeletsFeatureExtractor", "keep_candidates_percentage");
		((RealVariable) solution.getVariable(index)).setValue(0.1);

		activateRandomValueMaskingStrategy(solution, genomeHandler);
		return solution;
	}

	public static Solution activateRocket(final Solution solution, final GenomeHandler genomeHandler) {
		((BinaryVariable) solution.getVariable(genomeHandler.getIndexOfComponent("ROCKET"))).set(0, true);

		int index = genomeHandler.getIndexOfParameter("ROCKET", "n_kernels");
		((BinaryIntegerVariable) solution.getVariable(index)).setValue(100);

		activateRandomValueMaskingStrategy(solution, genomeHandler);
		return solution;
	}

	private static int getPositionInArray(final Object value, final Object... array) {
		for (int i = 0; i < array.length; i++) {
			if (array[i].equals(value)) {
				return i;
			}
		}
		return -1;
	}

}
