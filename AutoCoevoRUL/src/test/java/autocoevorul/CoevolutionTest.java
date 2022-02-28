package autocoevorul;

import static org.junit.Assert.assertNotNull;

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.execution.IDatasetSplitSet;
import org.junit.Ignore;
import org.junit.Test;

import com.google.common.eventbus.EventBus;

import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.featurerextraction.FeatureExtractionMoeaProblem;
import autocoevorul.featurerextraction.SolutionDecoding;
import autocoevorul.featurerextraction.genomehandler.BinaryAttributeSelectionIncludedGenomeHandler;
import autocoevorul.regression.RegressionGGPSolution;
import autocoevorul.regression.RegressionGgpProblem;
import autocoevorul.util.DataUtil;

public class CoevolutionTest extends AbstractTest {

	public CoevolutionTest() throws NoSuchMethodException, SecurityException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		super(BinaryAttributeSelectionIncludedGenomeHandler.class);
	}

	@Test
	@Ignore
	public void testRegressionSearchWithTsfresh() throws Exception {
		IDatasetSplitSet<ILabeledDataset<?>> datasetSplitSet = DataUtil.prepareDatasetSplits(this.getTestExperimentConfiguration(), new Random(this.getTestExperimentConfiguration().getSeed()));

		List<SolutionDecoding> validSolutionDecodings = new ArrayList<>();
		validSolutionDecodings.add(this.testGenomeHandler.decodeGenome(this.testGenomeHandler.activateTsfresh(this.testGenomeHandler.getEmptySolution(this.getFeatureExtractionMoeaProblem()))));

		List<List<Double>> groundTruthTest = new ArrayList<>();
		for (int fold = 0; fold < this.getTestExperimentConfiguration().getNumberOfFolds(); fold++) {
			groundTruthTest.add(datasetSplitSet.getFolds(fold).get(1).stream().map(instance -> (Double) instance.getLabel()).collect(Collectors.toList()));
		}

		FeatureExtractionMoeaProblem featureEvaluator = new FeatureExtractionMoeaProblem(new EventBus(), this.getTestExperimentConfiguration(), this.testGenomeHandler, datasetSplitSet);
		featureEvaluator.evaluateAll(validSolutionDecodings.stream().map(solutionDecoding -> solutionDecoding.getSolution()).collect(Collectors.toList()));

		RegressionGgpProblem runner = new RegressionGgpProblem(new EventBus(), this.getTestExperimentConfiguration(), validSolutionDecodings, groundTruthTest);
		RegressionGGPSolution best = runner.evaluateExtractors();

		assertNotNull(best);
	}

	@Test(expected = RuntimeException.class)
	public void testRegressionSearchWithEmptyListOfSolutions() throws Exception {
		ExperimentConfiguration experimentConfiguration = this.getTestExperimentConfiguration();

		List<SolutionDecoding> validSolutionDecodings = new ArrayList<>();
		List<List<Double>> groundTruthTest = new ArrayList<>();

		RegressionGgpProblem runner = new RegressionGgpProblem(new EventBus(), experimentConfiguration, validSolutionDecodings, groundTruthTest);
		runner.evaluateExtractors();
	}

}
