package autocoevorul;

import java.io.IOException;

import org.api4.java.ai.ml.core.dataset.serialization.DatasetDeserializationFailedException;
import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.BinaryIntegerVariable;
import org.moeaframework.core.variable.BinaryVariable;
import org.slf4j.Logger;

import com.google.common.eventbus.EventBus;

import ai.libs.jaicore.components.exceptions.ComponentNotFoundException;
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.featurerextraction.FeatureExtractionMoeaProblem;
import autocoevorul.featurerextraction.GenomeHandler;

public class AbstractTest {

	private static final Logger LOGGER = org.slf4j.LoggerFactory.getLogger(AbstractTest.class);

	private static final String CONFIG_FILE_PATH = "src/test/resources/searchspace/tests.cnf";
	protected static final String PYTHON_TEMPLATE_PATH = "../python_connection/run.py";

	private ExperimentConfiguration experimentConfiguration;

	protected GenomeHandler setupGenomeHandler() throws IOException, ComponentNotFoundException {
		return new GenomeHandler(this.getExperimentConfiguration());
	}

	protected ExperimentConfiguration getExperimentConfiguration() {
		if (this.experimentConfiguration == null) {
			this.experimentConfiguration = new ExperimentConfiguration(CONFIG_FILE_PATH);
		}
		return this.experimentConfiguration;
	}

	protected Solution getEmptySolution(final ExperimentConfiguration experimentConfiguration, final GenomeHandler genomeHandler) throws DatasetDeserializationFailedException {
		FeatureExtractionMoeaProblem problem = new FeatureExtractionMoeaProblem(new EventBus(), experimentConfiguration, genomeHandler, null);
		Solution solution = problem.newSolution();
		for (int i = 0; i < 65; i++) {
			((BinaryIntegerVariable) solution.getVariable(i)).setValue(this.getPositionInArray("False", "True", "False"));
		}
		((BinaryVariable) solution.getVariable(65)).set(0, false);
		((BinaryVariable) solution.getVariable(66)).set(0, false);
		((BinaryVariable) solution.getVariable(68)).set(0, false);
		return solution;
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
