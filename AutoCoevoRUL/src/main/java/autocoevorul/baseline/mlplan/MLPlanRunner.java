package autocoevorul.baseline.mlplan;

import java.io.IOException;
import java.sql.SQLException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.api4.java.ai.ml.core.dataset.splitter.SplitFailedException;
import org.api4.java.ai.ml.core.evaluation.IPrediction;
import org.api4.java.ai.ml.core.evaluation.IPredictionBatch;
import org.api4.java.ai.ml.core.exception.PredictionException;
import org.api4.java.ai.ml.core.exception.TrainingException;
import org.api4.java.algorithm.Timeout;

import com.google.common.eventbus.EventBus;

import ai.libs.jaicore.basic.ResourceFile;
import ai.libs.jaicore.components.exceptions.ComponentNotFoundException;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.ml.core.evaluation.evaluator.factory.MonteCarloCrossValidationEvaluatorFactory;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnTimeSeriesRegressionWrapper;
import ai.libs.mlplan.core.TasksAlreadyResolvedPathEvaluator;
import ai.libs.mlplan.sklearn.EMLPlanScikitLearnProblemType;
import ai.libs.mlplan.sklearn.MLPlan4ScikitLearn;
import ai.libs.mlplan.sklearn.builder.MLPlanScikitLearnBuilder;
import autocoevorul.AbstractRunner;
import autocoevorul.SearchResult;
import autocoevorul.baseline.randomsearch.PipelineEvaluationReport;
import autocoevorul.event.DatabaseWritingEventListener;
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.experiment.ICoevolutionConfig;

public class MLPlanRunner extends AbstractRunner {

	@Override
	public String getExperimentConfigurationFilePath() {
		return "conf/experiments/experiments.cnf";
	}

	@Override
	public String getDatabaseConfigurationFilePath() {
		return "conf/experiments/mlPlan.properties";
	}

	public static void main(final String[] args) throws ExperimentEvaluationFailedException, IOException, ComponentNotFoundException, ClassNotFoundException, InterruptedException,
			SplitFailedException, NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException {
		String executor = args[0];
		boolean setupDatabase = Boolean.parseBoolean(args[1]);
		boolean executeExperiments = Boolean.parseBoolean(args[2]);

		new MLPlanRunner().run(executor, setupDatabase, executeExperiments);
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

				MLPlanScikitLearnBuilder builder = new MLPlanScikitLearnBuilder(EMLPlanScikitLearnProblemType.RUL, true);
				builder.withNumCpus(experimentConfiguration.getNumCPUs());
				builder.withSearchSpaceConfigFile(new ResourceFile(experimentConfiguration.getRulSearchSpace()));
				builder.withPreferredNodeEvaluator(new TasksAlreadyResolvedPathEvaluator(Arrays.asList("AbstractRegressor", "BasicRegressor")));

				((MonteCarloCrossValidationEvaluatorFactory) builder.getLearnerEvaluationFactoryForSearchPhase()).withMeasure(experimentConfiguration.getPerformanceMeasure())
						.withNumMCIterations(experimentConfiguration.getNumberOfFolds()).withCacheSplitSets(true);
				((MonteCarloCrossValidationEvaluatorFactory) builder.getLearnerEvaluationFactoryForSelectionPhase()).withMeasure(experimentConfiguration.getPerformanceMeasure())
						.withNumMCIterations(5);

				builder.withTimeOut(experimentConfiguration.getTotalTimeout());
				builder.withNodeEvaluationTimeOut(new Timeout(experimentConfiguration.getRulTimeout().seconds() * 3, TimeUnit.SECONDS));
				builder.withCandidateEvaluationTimeOut(experimentConfiguration.getRulTimeout());
				builder.withSeed(experimentConfiguration.getSeed());

				MLPlan4ScikitLearn mlplan = new MLPlan4ScikitLearn(builder, experimentConfiguration.getTrainingData());
				mlplan.setPortionOfDataForPhase2(0f);
				mlplan.setLoggerName("mlplan");
				mlplan.registerListener(new DatabaseWritingEventListener(experimentConfiguration, dbConfig, executor));

				SearchResult searchResult = new SearchResult(new EventBus());
				ScikitLearnTimeSeriesRegressionWrapper<IPrediction, IPredictionBatch> optimizedRegressor = null;

				long start = System.currentTimeMillis();
				try {
					optimizedRegressor = (ScikitLearnTimeSeriesRegressionWrapper<IPrediction, IPredictionBatch>) mlplan.call();
					searchResult.update(new PipelineEvaluationReport(optimizedRegressor.toString(), "", mlplan.getInternalValidationErrorOfSelectedClassifier(), start, System.currentTimeMillis()));

				} catch (Exception e) {
					LOGGER.error("Building the classifier failed for pipeline {}", optimizedRegressor, e);
				}

				this.logFinalPipeline(experimentConfiguration, searchResult, experimentDBColumns, processor);
				this.executeFinalPipeline(experimentConfiguration, optimizedRegressor, experimentDBColumns, processor);

			} catch (IllegalArgumentException | SecurityException | IOException | TrainingException | PredictionException | SQLException e) {
				LOGGER.error("ML-Plan failed.", e);
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
