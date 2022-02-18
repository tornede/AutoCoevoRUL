package autocoevorul;

import static org.junit.Assert.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.nio.charset.StandardCharsets;

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
import org.junit.Test;
import org.moeaframework.core.Solution;

import com.google.common.hash.Hashing;

import ai.libs.jaicore.components.exceptions.ComponentNotFoundException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import autocoevorul.featurerextraction.SolutionDecoding;
import autocoevorul.featurerextraction.TimeseriesFeatureEngineeringScikitLearnWrapper;
import autocoevorul.featurerextraction.genomehandler.BinaryAttributeSelectionIncludedGenomeHandler;

public class BinaryAttributeSelectionIncludedGenomeHandlerTest extends AbstractTest{
	
	public BinaryAttributeSelectionIncludedGenomeHandlerTest() throws NoSuchMethodException, SecurityException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		super(BinaryAttributeSelectionIncludedGenomeHandler.class);
	}
	

	@Before
	public void purgeTmpDirectory() throws IOException {
		File tmpDirectory = new File("tmp/");
		if (tmpDirectory.isDirectory()) {
			FileUtils.deleteDirectory(tmpDirectory);
		}
	}
	
	@Test
	public void testCorrectGenomeSize() throws IOException, ComponentNotFoundException, ExperimentEvaluationFailedException {
		assertEquals( 
				24 // sensors
				+ 2 // AttributeFilter + 3 params
				+ 2 // Tsfresh
				, this.genomeHandler.getNumberOfVariables());
	}
	
	@Test
	public void testInitialTsfreshSolutions() throws ComponentNotFoundException {
		Solution tsfreshSolution = this.genomeHandler.activateTsfresh(this.genomeHandler.getEmptySolution(this.getFeatureExtractionMoeaProblem()));
		assertNotNull(tsfreshSolution);
		
		SolutionDecoding solutionDecoding = this.genomeHandler.decodeGenome(tsfreshSolution);
		assertNotNull(solutionDecoding);
		assertEquals("make_pipeline(TsfreshWrapper(tsfresh_features=[TsfreshFeature.MAXIMUM,TsfreshFeature.MINIMUM]))", solutionDecoding.getConstructionInstruction());
		assertEquals(3, solutionDecoding.getImports().split("\n").length);
		assertEquals("from ml4pdm.transformation.fixed_size import TsfreshFeature\n"
				+ "from ml4pdm.transformation.fixed_size import TsfreshWrapper\n"
				+ "from sklearn.pipeline import make_pipeline\n", solutionDecoding.getImports());
	}
	

	@Test
	public void testCorrectAmountOfFeaturesGeneratedByTSFresh()
			throws IOException, ComponentNotFoundException, DatasetDeserializationFailedException, TrainingException, PredictionException, InterruptedException, ExperimentEvaluationFailedException {
		Solution tsfreshSolution = this.genomeHandler.activateTsfresh(this.genomeHandler.getEmptySolution(this.getFeatureExtractionMoeaProblem()));
		SolutionDecoding solutionDecoding = this.genomeHandler.decodeGenome(tsfreshSolution);

		TimeseriesFeatureEngineeringScikitLearnWrapper<IPrediction, IPredictionBatch> sklearnWrapper = new TimeseriesFeatureEngineeringScikitLearnWrapper<>(solutionDecoding.getConstructionInstruction(), solutionDecoding.getImports());
		sklearnWrapper.setPythonTemplate(PYTHON_TEMPLATE_PATH);
		sklearnWrapper.setTimeout(this.getExperimentConfiguration().getFeatureCandidateTimeoutPerFold());
		sklearnWrapper.setSeed(1234);

		sklearnWrapper.fitAndPredict(this.getExperimentConfiguration().getTrainingData(), this.getExperimentConfiguration().getEvaluationData());

		String transformedTrainingDatasetName = this.getHashCodeForSolutionDecoding(solutionDecoding) + "_" + "CMAPSS_train_FD001.arff";
		ILabeledDataset<ILabeledInstance> transformedTrainingDataset = this.getExperimentConfiguration().readDataFile("tmp/tmp1/" + transformedTrainingDatasetName);
		assertEquals(this.getExperimentConfiguration().getTrainingData().getNumAttributes() * 2, transformedTrainingDataset.getNumAttributes());

		String transformedTestDatasetName = this.getHashCodeForSolutionDecoding(solutionDecoding) + "_" + "CMAPSS_train_FD001.arff";
		ILabeledDataset<ILabeledInstance> transformedTestDataset = this.getExperimentConfiguration().readDataFile("tmp/tmp1/" + transformedTestDatasetName);
		assertEquals(this.getExperimentConfiguration().getEvaluationData().getNumAttributes() * 2, transformedTestDataset.getNumAttributes());
	}

	public String getHashCodeForSolutionDecoding(final SolutionDecoding solutionDecoding) {
		String hashCode = Hashing.sha256().hashString(StringUtils.join(solutionDecoding.getConstructionInstruction(), solutionDecoding.getImports()), StandardCharsets.UTF_8).toString();
		return hashCode.startsWith("-") ? hashCode.replace("-", "1") : "0" + hashCode;
	}
	
}
