package autocoevorul;

import static org.junit.Assert.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.nio.charset.StandardCharsets;

import org.aeonbits.owner.ConfigFactory;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.api4.java.ai.ml.core.dataset.serialization.DatasetDeserializationFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.ai.ml.core.evaluation.IPrediction;
import org.api4.java.ai.ml.core.evaluation.IPredictionBatch;
import org.api4.java.ai.ml.core.exception.PredictionException;
import org.api4.java.ai.ml.core.exception.TrainingException;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.moeaframework.core.Solution;

import com.google.common.hash.Hashing;

import ai.libs.jaicore.components.exceptions.ComponentNotFoundException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.experiment.ICoevolutionConfig;
import autocoevorul.featurerextraction.SolutionDecoding;
import autocoevorul.featurerextraction.TimeseriesFeatureEngineeringScikitLearnWrapper;
import autocoevorul.featurerextraction.genomehandler.BinaryAttributeSelectionIncludedGenomeHandler;

public class BinaryAttributeSelectionIncludedGenomeHandlerTest extends AbstractTest {

	public BinaryAttributeSelectionIncludedGenomeHandlerTest() throws NoSuchMethodException, SecurityException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		super(BinaryAttributeSelectionIncludedGenomeHandler.class);
	}

	@Before
	public void purgeTmpDirectory() throws IOException {
		final File tmpDirectory = new File("tmp/");
		if (tmpDirectory.isDirectory()) {
			FileUtils.deleteDirectory(tmpDirectory);
		}
	}

	
	@Test
	public void testCorrectGenomeSize() throws IOException, ComponentNotFoundException, ExperimentEvaluationFailedException {
		assertEquals(24 // sensors
				+ 2 // AttributeFilter + 3 params
				+ 2 // Tsfresh
				, this.testGenomeHandler.getNumberOfVariables());
	}
	
	@Test
	public void testCorrectGenomeSizeWithMainSearchSpace() throws IOException, ComponentNotFoundException, ExperimentEvaluationFailedException {
		final String file = "src/test/resources/searchspace/tests.cnf";
		final ICoevolutionConfig config = (ICoevolutionConfig) ConfigFactory.create(ICoevolutionConfig.class).loadPropertiesFromFile(new File(file));
		config.setProperty("featureSearchspace", "searchspace/timeseries/timeseries_feature_extraction.json");
		final BinaryAttributeSelectionIncludedGenomeHandler genomeHandler = new BinaryAttributeSelectionIncludedGenomeHandler(
				new ExperimentConfiguration(file, config));

		assertEquals(72 // TsFresh
				+ 2 // AttributeTypes
				+ 1 // Filter
				+ 24 // Sensoren
				, genomeHandler.getNumberOfVariables());
	}


	@Test
	public void testInitialTsfreshSolutions() throws ComponentNotFoundException {
		final Solution tsfreshSolution = this.testGenomeHandler.activateTsfresh(this.testGenomeHandler.getEmptySolution(this.getFeatureExtractionMoeaProblem()));
		assertNotNull(tsfreshSolution);

		final SolutionDecoding solutionDecoding = this.testGenomeHandler.decodeGenome(tsfreshSolution);
		assertNotNull(solutionDecoding);
		assertEquals("make_pipeline(TsfreshWrapper(tsfresh_features=[TsfreshFeature.MAXIMUM,TsfreshFeature.MINIMUM]))", solutionDecoding.getConstructionInstruction());
		assertEquals(3, solutionDecoding.getImports().split("\n").length);
		assertEquals("from ml4pdm.transformation.fixed_size import TsfreshFeature\n" + "from ml4pdm.transformation.fixed_size import TsfreshWrapper\n" + "from sklearn.pipeline import make_pipeline\n", solutionDecoding.getImports());
	}


	@Test
	@Ignore
	public void testCorrectAmountOfFeaturesGeneratedByTSFresh()
			throws IOException, ComponentNotFoundException, DatasetDeserializationFailedException, TrainingException, PredictionException, InterruptedException, ExperimentEvaluationFailedException {
		final Solution tsfreshSolution = this.testGenomeHandler.activateTsfresh(this.testGenomeHandler.getEmptySolution(this.getFeatureExtractionMoeaProblem()));
		final SolutionDecoding solutionDecoding = this.testGenomeHandler.decodeGenome(tsfreshSolution);

		final TimeseriesFeatureEngineeringScikitLearnWrapper<IPrediction, IPredictionBatch> sklearnWrapper = new TimeseriesFeatureEngineeringScikitLearnWrapper<>(solutionDecoding.getConstructionInstruction(), solutionDecoding.getImports());
		sklearnWrapper.setPythonTemplate(PYTHON_TEMPLATE_PATH);
		sklearnWrapper.setTimeout(this.getTestExperimentConfiguration().getFeatureCandidateTimeoutPerFold());
		sklearnWrapper.setSeed(1234);

		sklearnWrapper.fitAndPredict(this.getTestExperimentConfiguration().getTrainingData(), this.getTestExperimentConfiguration().getEvaluationData());

		final String transformedTrainingDatasetName = this.getHashCodeForSolutionDecoding(solutionDecoding) + "_" + "CMAPSS_train_FD001.arff";
		final ILabeledDataset<ILabeledInstance> transformedTrainingDataset = this.getTestExperimentConfiguration().readDataFile("tmp/tmp1/" + transformedTrainingDatasetName);
		assertEquals(this.getTestExperimentConfiguration().getTrainingData().getNumAttributes() * 2, transformedTrainingDataset.getNumAttributes());

		final String transformedTestDatasetName = this.getHashCodeForSolutionDecoding(solutionDecoding) + "_" + "CMAPSS_train_FD001.arff";
		final ILabeledDataset<ILabeledInstance> transformedTestDataset = this.getTestExperimentConfiguration().readDataFile("tmp/tmp1/" + transformedTestDatasetName);
		assertEquals(this.getTestExperimentConfiguration().getEvaluationData().getNumAttributes() * 2, transformedTestDataset.getNumAttributes());
	}

	public String getHashCodeForSolutionDecoding(final SolutionDecoding solutionDecoding) {
		final String hashCode = Hashing.sha256().hashString(StringUtils.join(solutionDecoding.getConstructionInstruction(), solutionDecoding.getImports()), StandardCharsets.UTF_8).toString();
		return hashCode.startsWith("-") ? hashCode.replace("-", "1") : "0" + hashCode;
	}

}
