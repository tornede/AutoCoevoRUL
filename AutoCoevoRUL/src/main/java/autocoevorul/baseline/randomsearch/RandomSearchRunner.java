package autocoevorul.baseline.randomsearch;

import java.io.IOException;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutionException;

import org.api4.java.ai.ml.core.dataset.splitter.SplitFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.execution.IDatasetSplitSet;
import org.api4.java.ai.ml.core.exception.PredictionException;
import org.api4.java.ai.ml.core.exception.TrainingException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.moeaframework.core.PRNG;

import com.google.common.eventbus.EventBus;

import ai.libs.jaicore.components.exceptions.ComponentNotFoundException;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.timing.TimedComputation;
import autocoevorul.AbstractRunner;
import autocoevorul.SearchResult;
import autocoevorul.event.DatabaseWritingEventListener;
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.experiment.ICoevolutionConfig;
import autocoevorul.util.DataUtil;

public class RandomSearchRunner extends AbstractRunner {

	@Override
	public String getExperimentConfigurationFilePath() {
		return "conf/experiments/experiments.cnf";
	}

	@Override
	public String getDatabaseConfigurationFilePath() {
		return "conf/experiments/randomSearch.properties";
	}

	public static void main(final String[] args) throws ExperimentEvaluationFailedException, IOException, ComponentNotFoundException, ClassNotFoundException, InterruptedException, SplitFailedException, NoSuchFieldException,
			SecurityException, IllegalArgumentException, IllegalAccessException {
		String executor = args[0];
		boolean setupDatabase = Boolean.parseBoolean(args[1]);
		boolean executeExperiments = Boolean.parseBoolean(args[2]);

		new RandomSearchRunner().run(executor, setupDatabase, executeExperiments);
	}

	@Override
	public void executeExperiments(final String executor, final ICoevolutionConfig expConfig, final IDatabaseConfig dbConfig, final ExperimenterMySQLHandle handle) {
		IExperimentSetEvaluator evaluator = (final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) -> {
			try {
				Map<String, Object> experimentDBColumns = new HashMap<>();
				experimentDBColumns.put("executor", executor);
				processor.processResults(experimentDBColumns);
				experimentDBColumns.clear();

				ExperimentConfiguration experimentConfiguration = new ExperimentConfiguration(this.getExperimentConfigurationFilePath(), expConfig, experimentEntry);
				LOGGER.info("Running experiment with {}", experimentConfiguration);

				EventBus eventBus = new EventBus();
				eventBus.register(new DatabaseWritingEventListener(experimentConfiguration, dbConfig, executor));

				Random random = new Random(experimentConfiguration.getSeed());
				PRNG.setSeed(experimentConfiguration.getSeed());

				IDatasetSplitSet<ILabeledDataset<?>> datasetSplitSet = DataUtil.prepareDatasetSplits(experimentConfiguration, random);

				RandomSearch randomSearch = new RandomSearch(eventBus, experimentConfiguration, datasetSplitSet);
				PipelineEvaluationReport bestPipeline = TimedComputation.compute(() -> randomSearch.run(), experimentConfiguration.getTotalTimeout(), "Feature engineering interrupted");

				SearchResult searchResult = new SearchResult(eventBus, bestPipeline);
				this.logFinalPipeline(experimentConfiguration, searchResult, experimentDBColumns, processor);
				this.executeFinalPipeline(experimentConfiguration, searchResult, experimentDBColumns, processor);

			} catch (IllegalArgumentException | IllegalAccessException | NoSuchFieldException | SecurityException | SplitFailedException | IOException | TrainingException | PredictionException | SQLException | AlgorithmTimeoutedException
					| ExecutionException e) {
				LOGGER.error("Random search failed.", e);
			}
		};

		try {
			ExperimentRunner runner = new ExperimentRunner(expConfig, evaluator, handle);
			runner.sequentiallyConductExperiments(1);
		} catch (ExperimentDBInteractionFailedException | InterruptedException e) {
			LOGGER.error("Error trying to run experiments.", e);
			System.exit(1);
		}

	}

}
