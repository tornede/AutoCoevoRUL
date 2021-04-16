package autocoevorul.experiment;

import java.io.File;

import org.aeonbits.owner.ConfigFactory;
import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.experiments.ExperimentDatabasePreparer;
import ai.libs.jaicore.experiments.IExperimentSetConfig;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentAlreadyExistsInDatabaseException;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.IllegalExperimentSetupException;

public class Experimenter {

	private static Logger LOGGER = LoggerFactory.getLogger("experiment");

	private static boolean setupDatabase = true;

	private static void setupDatabase(final IExperimentSetConfig expConfig, final ExperimenterMySQLHandle handle) {
		try {
			handle.setup(expConfig);
		} catch (ExperimentDBInteractionFailedException e) {
			LOGGER.error("Couldn't setup the sql handle.", e);
			System.exit(1);
		}

		ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(expConfig, handle);
		try {
			preparer.synchronizeExperiments();
		} catch (ExperimentDBInteractionFailedException | IllegalExperimentSetupException | AlgorithmTimeoutedException | InterruptedException | AlgorithmExecutionCanceledException | ExperimentAlreadyExistsInDatabaseException e) {
			LOGGER.error("Couldn't synchrinze experiment table.", e);
			System.exit(1);
		}
	}

	public static void main(final String[] args) {
		ICoevolutionConfig expConfig = (ICoevolutionConfig) ConfigFactory.create(ICoevolutionConfig.class).loadPropertiesFromFile(new File("conf/experiments/experiments.cnf"));
		IDatabaseConfig dbConfig = (IDatabaseConfig) ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(new File("conf/experiments/coevolution.properties"));
		ExperimenterMySQLHandle handle = new ExperimenterMySQLHandle(dbConfig);

		if (setupDatabase) {
			setupDatabase(expConfig, handle);
		}

		// try {
		// ExperimentEvaluator evaluator = new ExperimentEvaluator(expConfig);
		// ExperimentRunner runner = new ExperimentRunner(expConfig, evaluator, handle, args[0]);
		// runner.sequentiallyConductExperiments(1);
		// } catch (ExperimentDBInteractionFailedException | InterruptedException e) {
		// LOGGER.error("Error trying to run experiments.", e);
		// System.exit(1);
		// }
	}

}
