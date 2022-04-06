package autocoevorul;

import static org.junit.Assert.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.nio.charset.StandardCharsets;

import org.apache.commons.io.FileUtils;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.ai.ml.core.exception.PredictionException;
import org.api4.java.ai.ml.core.exception.TrainingException;
import org.junit.Test;
import org.junit.jupiter.api.BeforeAll;
import org.moeaframework.core.Solution;

import com.google.common.hash.Hashing;

import ai.libs.jaicore.components.exceptions.ComponentNotFoundException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import autocoevorul.featurerextraction.ML4PdMTimeSeriesFeatureEngineeringWrapper;
import autocoevorul.featurerextraction.SolutionDecoding;
import autocoevorul.featurerextraction.genomehandler.BinaryAttributeSelectionIncludedGenomeHandler;

public class BinaryAttributeSelectionIncludedGenomeHandlerTest extends AbstractTest {

	public BinaryAttributeSelectionIncludedGenomeHandlerTest() throws NoSuchMethodException, SecurityException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		super(BinaryAttributeSelectionIncludedGenomeHandler.class);
	}

	@BeforeAll
	public void purgeTmpDirectory() throws IOException {
		final File tmpDirectory = new File("tmp/");
		if (tmpDirectory.isDirectory()) {
			FileUtils.deleteDirectory(tmpDirectory);
		}
	}

	@Test
	public void testCorrectGenomeSizeWithMainSearchSpace() throws IOException, ComponentNotFoundException, ExperimentEvaluationFailedException {
		assertEquals(72 // TsFresh
				+ 24 // Sensoren
				, this.testGenomeHandler.getNumberOfVariables());
	}

	private boolean executePipeline(final SolutionDecoding solutionDecoding) throws IOException, InterruptedException, TrainingException, PredictionException, ExperimentEvaluationFailedException {
		ML4PdMTimeSeriesFeatureEngineeringWrapper sklearnWrapper = new ML4PdMTimeSeriesFeatureEngineeringWrapper(solutionDecoding.getConstructionInstruction(), solutionDecoding.getImports());
		sklearnWrapper.setPythonTemplate(PYTHON_TEMPLATE_PATH);
		sklearnWrapper.setTimeout(this.getTestExperimentConfiguration().getFeatureCandidateTimeoutPerFold());
		sklearnWrapper.setSeed(1234);

		sklearnWrapper.fitAndPredict(this.getTestExperimentConfiguration().getTrainingData(), this.getTestExperimentConfiguration().getEvaluationData());
		return true;
	}

	@Test
	public void testInitialAttributeFilterSolutions() throws ComponentNotFoundException, TrainingException, PredictionException, IOException, InterruptedException, ExperimentEvaluationFailedException {
		Solution attributeFilterSolution = ((BinaryAttributeSelectionIncludedGenomeHandler) this.testGenomeHandler).activateAttributeFilter(this.testGenomeHandler.getEmptySolution(this.getFeatureExtractionMoeaProblem()));
		assertNotNull(attributeFilterSolution);

		SolutionDecoding solutionDecoding = this.testGenomeHandler.decodeGenome(attributeFilterSolution);
		assertNotNull(solutionDecoding);
		assertEquals("make_pipeline(AttributeFilter(attribute_ids=[1,2]))", solutionDecoding.getConstructionInstruction());
		assertEquals(2, solutionDecoding.getImports().split("\n").length);
		assertEquals("from ml4pdm.transformation import AttributeFilter\n" + "from sklearn.pipeline import make_pipeline\n", solutionDecoding.getImports());

		assertTrue(this.executePipeline(solutionDecoding));

		String transformedTrainingDatasetName = this.getHashCodeForSolutionDecoding(solutionDecoding) + "_" + "CMAPSS_FD001_train.pdmff";
		ILabeledDataset<ILabeledInstance> transformedTrainingDataset = this.getTestExperimentConfiguration().readDataFile("tmp/tmp1/" + transformedTrainingDatasetName);
		assertEquals(2, transformedTrainingDataset.getNumAttributes());

		String transformedTestDatasetName = this.getHashCodeForSolutionDecoding(solutionDecoding) + "_" + "CMAPSS_FD001_train.pdmff";
		ILabeledDataset<ILabeledInstance> transformedTestDataset = this.getTestExperimentConfiguration().readDataFile("tmp/tmp1/" + transformedTestDatasetName);
		assertEquals(2, transformedTestDataset.getNumAttributes());

	}

	@Test
	public void testInitialTsfreshSolutions() throws ComponentNotFoundException, TrainingException, PredictionException, IOException, InterruptedException, ExperimentEvaluationFailedException {
		Solution tsfreshSolution = this.testGenomeHandler.activateTsfresh(this.testGenomeHandler.getEmptySolution(this.getFeatureExtractionMoeaProblem()));
		assertNotNull(tsfreshSolution);

		SolutionDecoding solutionDecoding = this.testGenomeHandler.decodeGenome(tsfreshSolution);
		assertNotNull(solutionDecoding);
		assertEquals("make_pipeline(TsfreshWrapper(tsfresh_features=[TsfreshFeature.MAXIMUM,TsfreshFeature.MINIMUM]))", solutionDecoding.getConstructionInstruction());
		assertEquals(3, solutionDecoding.getImports().split("\n").length);
		assertEquals("from ml4pdm.transformation.fixed_size._tsfresh import TsfreshFeature\n" //
				+ "from ml4pdm.transformation.fixed_size._tsfresh import TsfreshWrapper\n" //
				+ "from sklearn.pipeline import make_pipeline\n", //
				solutionDecoding.getImports());

		assertTrue(this.executePipeline(solutionDecoding));

		String transformedTrainingDatasetName = this.getHashCodeForSolutionDecoding(solutionDecoding) + "_" + "CMAPSS_FD001_train.pdmff";
		ILabeledDataset<ILabeledInstance> transformedTrainingDataset = this.getTestExperimentConfiguration().readDataFile("tmp/tmp1/" + transformedTrainingDatasetName);
		assertEquals(48, transformedTrainingDataset.getNumAttributes());

		String transformedTestDatasetName = this.getHashCodeForSolutionDecoding(solutionDecoding) + "_" + "CMAPSS_FD001_train.pdmff";
		ILabeledDataset<ILabeledInstance> transformedTestDataset = this.getTestExperimentConfiguration().readDataFile("tmp/tmp1/" + transformedTestDatasetName);
		assertEquals(48, transformedTestDataset.getNumAttributes());
	}

	public String getHashCodeForSolutionDecoding(final SolutionDecoding solutionDecoding) {
		final String hashCode = Hashing.sha256().hashString(solutionDecoding.getConstructionInstruction(), StandardCharsets.UTF_8).toString();
		return hashCode.startsWith("-") ? hashCode.replace("-", "1") : "0" + hashCode;
	}

}
