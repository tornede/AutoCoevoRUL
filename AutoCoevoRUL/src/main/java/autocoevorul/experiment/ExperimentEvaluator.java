package autocoevorul.experiment;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentFailurePredictionException;

public class ExperimentEvaluator implements IExperimentSetEvaluator {

	private static final Logger LOGGER = LoggerFactory.getLogger("experiment");
	private static final String PYTHON_TEMPLATE_PATH = "../python_connection/run.py";
	private ICoevolutionConfig fullExperimentConfiguration;

	public ExperimentEvaluator(final ICoevolutionConfig fullExperimentConfiguration) {
		super();
		this.fullExperimentConfiguration = fullExperimentConfiguration;
	}

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException, ExperimentFailurePredictionException, InterruptedException {
		// TODO
		// ExperimentConfiguration experimentConfiguration = new ExperimentConfiguration(experimentEntry, this.fullExperimentConfiguration);
		//
		// Map<String, Object> map = new HashMap<>();
		// map.put("tsFE", experimentConfiguration.getTimeseriesTransformer());
		// map.put("regressor", experimentConfiguration.getRegressor());
		// processor.processResults(map);
		// map.clear();
		//
		// try {
		// ScikitLearnWrapper<SingleTargetRegressionPrediction, SingleTargetRegressionPredictionBatch> model = new ScikitLearnWrapper<SingleTargetRegressionPrediction, SingleTargetRegressionPredictionBatch>(
		// experimentConfiguration.getPipeline(), experimentConfiguration.getImports(), false, experimentConfiguration.getProblemType().getSkLearnProblemType());
		// model.setPythonTemplate(PYTHON_TEMPLATE_PATH);
		// model.setLoggerName("experiment");
		// model.setSeed(experimentConfiguration.getNumberOfSeeds());
		// model.fitAndPredict(experimentConfiguration.getTrainingData(), experimentConfiguration.getEvaluationData());
		//
		// SupervisedLearnerExecutor executor = new SupervisedLearnerExecutor();
		// executor.setLoggerName("experiment");
		// ILearnerRunReport report = executor.execute(model, experimentConfiguration.getTrainingData(), experimentConfiguration.getEvaluationData());
		// List<Double> expected = (List<Double>) report.getPredictionDiffList().getGroundTruthAsList();
		// List<IRegressionPrediction> predicted = (List<IRegressionPrediction>) report.getPredictionDiffList().getPredictionsAsList();
		//
		// for (ERulPerformanceMeasure performanceMeasure : ERulPerformanceMeasure.values()) {
		// map.put("performance_" + performanceMeasure.name().toLowerCase(), performanceMeasure.loss(expected, predicted));
		// }
		//
		// processor.processResults(map);
		// map.clear();
		// LOGGER.info("Finished Experiment {}. Results: {}", experimentEntry.getExperiment().getValuesOfKeyFields(), map);
		//
		// } catch (Exception e) {
		// LOGGER.info("Evaluating the classifier failed for pipeline {}", experimentConfiguration.getPipeline(), e);
		// throw new ExperimentEvaluationFailedException(e);
		// }
	}

}
