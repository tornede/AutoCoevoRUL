package autocoevorul;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.api4.java.ai.ml.core.dataset.splitter.SplitFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.execution.IDatasetSplitSet;
import org.api4.java.ai.ml.core.exception.PredictionException;
import org.api4.java.ai.ml.core.exception.TrainingException;
import org.api4.java.common.attributedobjects.ObjectEvaluationFailedException;
import org.junit.Test;

import com.google.common.eventbus.EventBus;

import ai.libs.jaicore.basic.ResourceFile;
import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.components.api.IComponentRepository;
import ai.libs.jaicore.components.exceptions.ComponentNotFoundException;
import ai.libs.jaicore.components.model.Component;
import ai.libs.jaicore.components.model.ComponentUtil;
import ai.libs.jaicore.components.model.NumericParameterDomain;
import ai.libs.jaicore.components.model.Parameter;
import ai.libs.jaicore.components.serialization.ComponentSerialization;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import autocoevorul.featurerextraction.SolutionDecoding;
import autocoevorul.featurerextraction.genomehandler.BinaryAttributeSelectionIncludedGenomeHandler;
import autocoevorul.regression.CompletePipelineEvaluator;
import autocoevorul.util.DataUtil;

public class CompetePipelineEvaluatorTest extends AbstractTest {

	public CompetePipelineEvaluatorTest() throws NoSuchMethodException, SecurityException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		super(BinaryAttributeSelectionIncludedGenomeHandler.class);
	}

	@Test
	public void test() throws ComponentNotFoundException, IllegalAccessException, IllegalArgumentException, NoSuchFieldException, SecurityException, ExperimentEvaluationFailedException, InterruptedException, SplitFailedException,
			TrainingException, PredictionException, IOException, ObjectEvaluationFailedException {
		IDatasetSplitSet<ILabeledDataset<?>> datasetSplitSet = DataUtil.prepareDatasetSplits(this.getTestExperimentConfiguration(), new Random(this.getTestExperimentConfiguration().getSeed()));

		List<SolutionDecoding> featureSolutionDecodings = new ArrayList<>();
		featureSolutionDecodings.add(this.testGenomeHandler.decodeGenome(this.testGenomeHandler.activateTsfresh(this.testGenomeHandler.getEmptySolution(this.getFeatureExtractionMoeaProblem()))));

		for (int i = 0; i < datasetSplitSet.getNumberOfSplits(); i++) {
			this.executePipeline(featureSolutionDecodings.get(0), datasetSplitSet.getFolds(i).get(0), datasetSplitSet.getFolds(i).get(1));
		}

		List<List<Double>> groundTruthTest = new ArrayList<>();
		for (int fold = 0; fold < this.getTestExperimentConfiguration().getNumberOfFolds(); fold++) {
			groundTruthTest.add(datasetSplitSet.getFolds(fold).get(1).stream().map(instance -> (Double) instance.getLabel()).collect(Collectors.toList()));
		}

		IComponentRepository componentRepository = new ComponentSerialization().deserializeRepository(new ResourceFile(this.getTestExperimentConfiguration().getRegressionSearchpace()));
		Component componentToCompose = new Component("ComponentToSearch");
		componentToCompose.addProvidedInterface("ComponentToSearch");
		componentToCompose.addParameter(new Parameter("feature_extractor_id", new NumericParameterDomain(true, 0, featureSolutionDecodings.size() - 1), 0));
		componentToCompose.addRequiredInterface(this.getTestExperimentConfiguration().getRegressionRequiredInterface(), this.getTestExperimentConfiguration().getRegressionRequiredInterface());
		componentRepository.add(componentToCompose);

		IComponentInstance randomComponentInstance = ComponentUtil.getRandomInstantiationOfComponent("ComponentToSearch", componentRepository, new Random(42));

		Double performance = new CompletePipelineEvaluator(new EventBus(), this.getTestExperimentConfiguration(), featureSolutionDecodings, groundTruthTest).evaluate(randomComponentInstance);
		assertTrue(performance > 0.98);
	}

}
