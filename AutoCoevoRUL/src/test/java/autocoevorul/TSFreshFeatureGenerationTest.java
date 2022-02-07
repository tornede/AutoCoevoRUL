package autocoevorul;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Comparator;
import java.util.List;

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
import org.moeaframework.core.variable.BinaryIntegerVariable;

import com.google.common.hash.Hashing;

import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.components.api.IParameter;
import ai.libs.jaicore.components.exceptions.ComponentNotFoundException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnTimeSeriesFeatureEngineeringWrapper;
import autocoevorul.featurerextraction.GenomeHandler;
import autocoevorul.featurerextraction.SolutionDecoding;

public class TSFreshFeatureGenerationTest extends AbstractTest {

	@Before
	public void purgeTmpDirectory() throws IOException {
		File tmpDirectory = new File("tmp/");
		if (tmpDirectory.isDirectory()) {
			FileUtils.deleteDirectory(tmpDirectory);
		}
	}

	@Test
	public void testCorrectAmountOfFeaturesGeneratedByTSFresh()
			throws IOException, ComponentNotFoundException, DatasetDeserializationFailedException, TrainingException, PredictionException, InterruptedException, ExperimentEvaluationFailedException {
		GenomeHandler genomeHandler = new GenomeHandler(this.getExperimentConfiguration());
		Solution solution = this.getEmptySolution(this.getExperimentConfiguration(), genomeHandler);

		// set minimum = true
		((BinaryIntegerVariable) solution.getVariable(0)).setValue(this.getPositionInArray("True", "True", "False"));

		// set has_duplicate = true
		((BinaryIntegerVariable) solution.getVariable(59)).setValue(this.getPositionInArray("True", "True", "False"));

		SolutionDecoding solutionDecoding = genomeHandler.decodeGenome(solution);
		List<IComponentInstance> componentInstances = solutionDecoding.getComponentInstances();
		componentInstances.sort(new Comparator<IComponentInstance>() {
			@Override
			public int compare(final IComponentInstance c1, final IComponentInstance c2) {
				return c1.getComponent().getName().compareTo(c2.getComponent().getName());
			}
		});

		assertEquals(1, componentInstances.size());

		IComponentInstance rootComponentInstance = componentInstances.get(0);
		assertEquals("python_connection.feature_generation.tsfresh_feature_generator.TsFreshBasedFeatureExtractor", rootComponentInstance.getComponent().getName());

		IComponentInstance dictionaryComponentInstance = rootComponentInstance.getSatisfactionOfRequiredInterface("default_fc_parameters").get(0);
		assertEquals("python_connection.feature_generation.tsfresh_feature_generator.FCParametersDictionary", dictionaryComponentInstance.getComponent().getName());

		for (IParameter parameter : dictionaryComponentInstance.getComponent().getParameters()) {
			if (parameter.getName().equals("minimum") || parameter.getName().equals("has_duplicate")) {
				assertEquals("True", dictionaryComponentInstance.getParameterValue(parameter.getName()));
			} else {
				assertEquals("False", dictionaryComponentInstance.getParameterValue(parameter.getName()));
			}
		}

		System.out.println(solutionDecoding.getConstructionInstruction());
		ScikitLearnTimeSeriesFeatureEngineeringWrapper<IPrediction, IPredictionBatch> sklearnWrapper = new ScikitLearnTimeSeriesFeatureEngineeringWrapper<>(solutionDecoding.getConstructionInstruction(), solutionDecoding.getImports());
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
