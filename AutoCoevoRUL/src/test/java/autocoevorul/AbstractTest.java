package autocoevorul;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.Random;

import org.api4.java.ai.ml.core.dataset.serialization.DatasetDeserializationFailedException;
import org.api4.java.ai.ml.core.dataset.splitter.SplitFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.execution.IDatasetSplitSet;
import org.moeaframework.core.PRNG;
import org.slf4j.Logger;

import com.google.common.eventbus.EventBus;

import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.featurerextraction.FeatureExtractionMoeaProblem;
import autocoevorul.featurerextraction.genomehandler.GenomeHandler;
import autocoevorul.util.DataUtil;

public abstract class AbstractTest {

	private static final Logger LOGGER = org.slf4j.LoggerFactory.getLogger(AbstractTest.class);

	private static final String TEST_CONFIG_FILE_PATH = "src/test/resources/searchspace/tests.cnf";

	protected static final String PYTHON_TEMPLATE_PATH = "src/main/resources/ml4pdm.py";

	private ExperimentConfiguration testExperimentConfiguration;

	protected GenomeHandler testGenomeHandler;

	public AbstractTest(final Class<? extends GenomeHandler> genomeHandlerClass) throws NoSuchMethodException, SecurityException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		final Constructor<?> cons = genomeHandlerClass.getConstructor(ExperimentConfiguration.class);
		this.testGenomeHandler = (GenomeHandler) cons.newInstance(this.getTestExperimentConfiguration());
	}

	protected ExperimentConfiguration getTestExperimentConfiguration() {
		if (this.testExperimentConfiguration == null) {
			this.testExperimentConfiguration = new ExperimentConfiguration(TEST_CONFIG_FILE_PATH);
		}
		return this.testExperimentConfiguration;
	}

	protected FeatureExtractionMoeaProblem getFeatureExtractionMoeaProblem() {
		try {
			final ExperimentConfiguration experimentConfiguration = this.getTestExperimentConfiguration();
			final Random random = new Random(1);
			PRNG.setSeed(experimentConfiguration.getSeed());
			IDatasetSplitSet<ILabeledDataset<?>> datasetSplitSet;
			datasetSplitSet = DataUtil.prepareDatasetSplits(experimentConfiguration, random);
			return new FeatureExtractionMoeaProblem(new EventBus(), this.getTestExperimentConfiguration(), this.testGenomeHandler, datasetSplitSet);
		} catch (DatasetDeserializationFailedException | IllegalAccessException | IllegalArgumentException | NoSuchFieldException | SecurityException | ExperimentEvaluationFailedException | InterruptedException | SplitFailedException e) {
			throw new RuntimeException();
		}
	}

	protected int getPositionInArray(final Object value, final Object... array) {
		for (int i = 0; i < array.length; i++) {
			if (array[i].equals(value)) {
				return i;
			}
		}
		return -1;
	}

}
