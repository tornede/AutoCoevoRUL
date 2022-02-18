package autocoevorul.featurerextraction.genomehandler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.BinaryIntegerVariable;
import org.moeaframework.core.variable.BinaryVariable;
import org.moeaframework.core.variable.RealVariable;

import ai.libs.jaicore.components.api.IComponent;
import ai.libs.jaicore.components.exceptions.ComponentNotFoundException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.featurerextraction.FeatureExtractionMoeaProblem;

public class BasicGenomeHandler extends GenomeHandler{

	public BasicGenomeHandler(final ExperimentConfiguration experimentConfiguration) throws IOException, ComponentNotFoundException, ExperimentEvaluationFailedException {
		super(experimentConfiguration);
	}

	@Override
	protected void setupAdditionalComponents(
			ExperimentConfiguration experimentConfiguration) throws ExperimentEvaluationFailedException {
		// nothing to do here		
	}
	
	@Override
	public List<Solution> getInitialSolutions(final FeatureExtractionMoeaProblem problem, final int populationSize) throws ComponentNotFoundException {
		List<Solution> initialSolutions = new ArrayList<>();

		// tsfresh only
		Solution solution = this.getEmptySolution(problem);
		this.activateTsfresh(solution);
		initialSolutions.add(solution);

		// UltrafastShapelet only
		solution = this.getEmptySolution(problem);
		this.activateUltraFastShapelet(solution);
		initialSolutions.add(solution);

		// Rocket only
		solution = this.getEmptySolution(problem);
		this.activateRocket(solution);
		initialSolutions.add(solution);

		// tsfresh + UltrafastShapelet
		solution = this.getEmptySolution(problem);
		this.activateTsfresh(solution);
		this.activateUltraFastShapelet(solution);
		initialSolutions.add(solution);

		// tsfresh + Rocket
		solution = this.getEmptySolution(problem);
		this.activateTsfresh(solution);
		this.activateRocket(solution);
		initialSolutions.add(solution);

		// UltrafastShapelet + Rocket
		solution = this.getEmptySolution(problem);
		this.activateUltraFastShapelet(solution);
		this.activateRocket(solution);
		initialSolutions.add(solution);

		// tsfresh + UltrafastShapelet + Rocket
		solution = this.getEmptySolution(problem);
		this.activateTsfresh(solution);
		this.activateUltraFastShapelet(solution);
		this.activateRocket(solution);
		initialSolutions.add(solution);

		return initialSolutions;
	}
	
	@Override
	public Solution getEmptySolution(final FeatureExtractionMoeaProblem problem) {
		Solution solution = problem.newSolution();
		// ((BinaryVariable) solution.getVariable(genomeHandler.getIndexOfComponent("FCParametersDictionary"))).set(0, false);
		for (Integer index : this.getIndexOfParameters("TsfreshFeature")) {
			((BinaryIntegerVariable) solution.getVariable(index)).setValue(getPositionInArray("False", new String[]{"True", "False"}));
		}

		((BinaryVariable) solution.getVariable(this.getIndexOfComponent("UltraFastShapeletsFeatureExtractor"))).set(0, false);
		((BinaryVariable) solution.getVariable(this.getIndexOfComponent("ROCKET"))).set(0, false);

		// as RealVariables are initialized with NaN, we should set this to some value here to avoid problems even if it is not used
		int index = this.getIndexOfParameter("UltraFastShapeletsFeatureExtractor", "keep_candidates_percentage");
		((RealVariable) solution.getVariable(index)).setValue(0.1);

		return solution;
	}
	
	@Override
	public Solution activateTsfresh(Solution solution) {
		String[] parameterName = new String[] { "has_duplicate_max", "binned_entropy", "last_location_of_maximum", "abs_energy", "c3", "value_count", "mean_second_derivative_central",
				"first_location_of_minimum", "standard_deviation", "length", "mean_abs_change", "has_duplicate_min", "mean_change", "sum_values",
				"percentage_of_reoccurring_datapoints_to_all_datapoints", "range_count", "absolute_sum_of_changes", "energy_ratio_by_chunks", "last_location_of_minimum", "linear_trend",
				"variance_larger_than_standard_deviation", "spkt_welch_density", "cid_ce", "symmetry_looking", "has_duplicate", "skewness", "count_above", "count_above_mean",
				"longest_strike_below_mean", "mean", "agg_autocorrelation", "ratio_value_number_to_time_series_length", "fft_aggregated", "first_location_of_maximum", "partial_autocorrelation",
				"sum_of_reoccurring_data_points", "count_below", "count_below_mean", "variance", "longest_strike_above_mean", "median", "kurtosis", "minimum", "time_reversal_asymmetry_statistic",
				"number_crossing_m", "sum_of_reoccurring_values", "maximum" };
		for (int i = 0; i < parameterName.length; i++) {
			int index = this.getIndexOfParameter("FCParametersDictionary", parameterName[i]);
			((BinaryIntegerVariable) solution.getVariable(index)).setValue(getPositionInArray("True", new String[]{"True", "False"}));
		}
		activateRandomValueMaskingStrategy(solution);
		return solution;
	}

	private Solution activateRandomValueMaskingStrategy(final Solution solution) {
		((BinaryVariable) solution.getVariable(this.getIndexOfComponent("RandomValueMaskingStrategy"))).set(0, true);
		return solution;
	}

	private Solution activateUltraFastShapelet(final Solution solution) {
		((BinaryVariable) solution.getVariable(this.getIndexOfComponent("UltraFastShapeletsFeatureExtractor"))).set(0, true);

		int index = this.getIndexOfParameter("UltraFastShapeletsFeatureExtractor", "keep_candidates_percentage");
		((RealVariable) solution.getVariable(index)).setValue(0.1);

		activateRandomValueMaskingStrategy(solution);
		return solution;
	}

	private Solution activateRocket(final Solution solution) {
		((BinaryVariable) solution.getVariable(this.getIndexOfComponent("ROCKET"))).set(0, true);

		int index = this.getIndexOfParameter("ROCKET", "n_kernels");
		((BinaryIntegerVariable) solution.getVariable(index)).setValue(100);

		activateRandomValueMaskingStrategy(solution);
		return solution;
	}

}
