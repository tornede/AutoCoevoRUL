package autocoevorul;

import static org.junit.Assert.assertNotNull;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.api4.java.ai.ml.core.dataset.serialization.DatasetDeserializationFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.execution.IDatasetSplitSet;
import org.junit.Test;
import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.BinaryIntegerVariable;
import org.moeaframework.core.variable.BinaryVariable;

import com.google.common.eventbus.EventBus;

import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.featurerextraction.FeatureExtractionMoeaProblem;
import autocoevorul.featurerextraction.GenomeHandler;
import autocoevorul.featurerextraction.SolutionDecoding;
import autocoevorul.regression.RegressionGGPSolution;
import autocoevorul.regression.RegressionGgpProblem;
import autocoevorul.util.DataUtil;

public class CoevolutionTest extends AbstractTest {

	private Solution getRocketComponentInstanceSolutionToTest(final ExperimentConfiguration experimentConfiguration, final GenomeHandler genomeHandler) throws DatasetDeserializationFailedException {
		Solution solution = this.getEmptySolution(experimentConfiguration, genomeHandler);
		((BinaryVariable) solution.getVariable(65)).set(0, true);
		((BinaryVariable) solution.getVariable(66)).set(0, true);
		((BinaryIntegerVariable) solution.getVariable(67)).setValue(10);

		return solution;
	}

	@Test
	public void testRegressionSearchWithRocket() throws Exception {
		ExperimentConfiguration experimentConfiguration = this.getExperimentConfiguration();
		GenomeHandler genomeHandler = this.setupGenomeHandler();
		IDatasetSplitSet<ILabeledDataset<?>> datasetSplitSet = DataUtil.prepareDatasetSplits(experimentConfiguration, new Random(experimentConfiguration.getSeed()));

		List<SolutionDecoding> validSolutionDecodings = new ArrayList<>();
		validSolutionDecodings.add(genomeHandler.decodeGenome(this.getRocketComponentInstanceSolutionToTest(experimentConfiguration, genomeHandler)));

		List<List<Double>> groundTruthTest = new ArrayList<>();
		for (int fold = 0; fold < experimentConfiguration.getNumberOfFolds(); fold++) {
			groundTruthTest.add(datasetSplitSet.getFolds(fold).get(1).stream().map(instance -> (Double) instance.getLabel()).collect(Collectors.toList()));
		}

		FeatureExtractionMoeaProblem featureEvaluator = new FeatureExtractionMoeaProblem(new EventBus(), experimentConfiguration, genomeHandler, datasetSplitSet);
		featureEvaluator.evaluateAll(validSolutionDecodings.stream().map(solutionDecoding -> solutionDecoding.getSolution()).collect(Collectors.toList()));

		RegressionGgpProblem runner = new RegressionGgpProblem(new EventBus(), experimentConfiguration, validSolutionDecodings, groundTruthTest);
		RegressionGGPSolution best = runner.evaluateExtractors();

		assertNotNull(best);
	}

	private Solution getTsfreshComponentInstanceSolutionToTest(final ExperimentConfiguration experimentConfiguration, final GenomeHandler genomeHandler) throws DatasetDeserializationFailedException {
		Solution solution = this.getEmptySolution(experimentConfiguration, genomeHandler);
		((BinaryIntegerVariable) solution.getVariable(0)).setValue(this.getPositionInArray("True", "True", "False"));
		return solution;
	}

	@Test
	public void testRegressionSearchWithTsfresh() throws Exception {
		ExperimentConfiguration experimentConfiguration = this.getExperimentConfiguration();
		GenomeHandler genomeHandler = this.setupGenomeHandler();
		IDatasetSplitSet<ILabeledDataset<?>> datasetSplitSet = DataUtil.prepareDatasetSplits(experimentConfiguration, new Random(experimentConfiguration.getSeed()));

		List<SolutionDecoding> validSolutionDecodings = new ArrayList<>();
		validSolutionDecodings.add(genomeHandler.decodeGenome(this.getTsfreshComponentInstanceSolutionToTest(experimentConfiguration, genomeHandler)));

		List<List<Double>> groundTruthTest = new ArrayList<>();
		for (int fold = 0; fold < experimentConfiguration.getNumberOfFolds(); fold++) {
			groundTruthTest.add(datasetSplitSet.getFolds(fold).get(1).stream().map(instance -> (Double) instance.getLabel()).collect(Collectors.toList()));
		}

		FeatureExtractionMoeaProblem featureEvaluator = new FeatureExtractionMoeaProblem(new EventBus(), experimentConfiguration, genomeHandler, datasetSplitSet);
		featureEvaluator.evaluateAll(validSolutionDecodings.stream().map(solutionDecoding -> solutionDecoding.getSolution()).collect(Collectors.toList()));

		RegressionGgpProblem runner = new RegressionGgpProblem(new EventBus(), experimentConfiguration, validSolutionDecodings, groundTruthTest);
		RegressionGGPSolution best = runner.evaluateExtractors();

		assertNotNull(best);
	}

	@Test(expected = RuntimeException.class)
	public void testRegressionSearchWithEmptyListOfSolutions() throws Exception {
		ExperimentConfiguration experimentConfiguration = this.getExperimentConfiguration();

		List<SolutionDecoding> validSolutionDecodings = new ArrayList<>();
		List<List<Double>> groundTruthTest = new ArrayList<>();

		RegressionGgpProblem runner = new RegressionGgpProblem(new EventBus(), experimentConfiguration, validSolutionDecodings, groundTruthTest);
		runner.evaluateExtractors();
	}

}
