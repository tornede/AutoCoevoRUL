package autocoevorul.baseline.randomsearch;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalDouble;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;

import org.apache.commons.lang.exception.ExceptionUtils;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.IPrediction;
import org.api4.java.ai.ml.core.evaluation.IPredictionBatch;
import org.api4.java.ai.ml.core.evaluation.execution.IDatasetSplitSet;
import org.api4.java.ai.ml.regression.evaluation.IRegressionPrediction;
import org.api4.java.algorithm.Timeout;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.eventbus.EventBus;

import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.ml.regression.singlelabel.SingleTargetRegressionPrediction;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnWrapper;
import ai.libs.jaicore.timing.TimedComputation;
import ai.libs.mlplan.sklearn.AScikitLearnLearnerFactory;
import autocoevorul.SearchResult;
import autocoevorul.event.RegressorEvaluatedEvent;
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.util.ScikitLearnUtil;

public abstract class AbstractCompletePipelineEvaluator implements Runnable {

	private static Logger LOGGER = LoggerFactory.getLogger(AbstractCompletePipelineEvaluator.class);

	private EventBus eventBus;

	protected final ExperimentConfiguration experimentConfiguration;
	protected final IDatasetSplitSet<ILabeledDataset<?>> datasetSplitSet;
	private final List<List<? extends Double>> groundTruthsForSplits;
	private final Timeout timeout;
	private IComponentInstance componentInstance;
	private SearchResult bestPipeline;

	public AbstractCompletePipelineEvaluator(final EventBus eventBus, final ExperimentConfiguration experimentConfiguration, final IDatasetSplitSet<ILabeledDataset<?>> datasetSplitSet,
			final Timeout timeout, final IComponentInstance componentInstance, final SearchResult bestPipeline) {
		super();
		this.eventBus = eventBus;
		this.experimentConfiguration = experimentConfiguration;
		this.datasetSplitSet = datasetSplitSet;
		this.timeout = timeout;
		this.componentInstance = componentInstance;

		this.groundTruthsForSplits = new ArrayList<>(experimentConfiguration.getNumberOfFolds());
		for (int split = 0; split < datasetSplitSet.getNumberOfSplits(); split++) {
			this.groundTruthsForSplits.add(this.datasetSplitSet.getFolds(split).get(1).stream().map(instance -> (Double) instance.getLabel()).collect(Collectors.toList()));
		}
		this.bestPipeline = bestPipeline;
	}

	@Override
	public void run() {
		Pair<String, String> constructionInstructionAndImports = ScikitLearnUtil.createConstructionInstructionAndImportsFromComponentInstance(this.componentInstance);
		String constructionInstruction = constructionInstructionAndImports.getX();
		String imports = constructionInstructionAndImports.getY();

		List<Double> performances = new ArrayList<>(this.experimentConfiguration.getNumberOfFolds());
		String exception = "";
		long trainStart = System.currentTimeMillis();
		List<Long> runtimes = new ArrayList<>();
		for (int i = 0; i < this.experimentConfiguration.getNumberOfFolds(); i++) {

			try {
				long foldStart = System.currentTimeMillis();
				if (Thread.interrupted()) {
					throw new InterruptedException();
				}

				ScikitLearnWrapper<IPrediction, IPredictionBatch> learner = this.setupScikitlearnWrapper(constructionInstruction, imports, this.timeout);
				IPredictionBatch predictionsForSplit = this.runScikitLearnWrapper(learner, i, this.timeout);

				List<IRegressionPrediction> doublePredictionsForSplit = predictionsForSplit.getPredictions().stream()
						.map(prediction -> new SingleTargetRegressionPrediction(Math.max(0, ((IRegressionPrediction) prediction).getDoublePrediction()))).collect(Collectors.toList());
				performances.add(this.experimentConfiguration.getPerformanceMeasure().loss(this.groundTruthsForSplits.get(i), doublePredictionsForSplit));
				runtimes.add(System.currentTimeMillis() - foldStart);
			} catch (IOException | AlgorithmTimeoutedException | ExecutionException e) {
				exception = ExceptionUtils.getStackTrace(e);
				LOGGER.warn("Failed to fit and predict with Scikitlearn wrapper with pipeline {} on dataset {}. Error: \n {}", constructionInstruction, this.experimentConfiguration.getDatasetName(),
						ExceptionUtils.getStackTrace(e));
				break;
			} catch (InterruptedException e) {
				exception = ExceptionUtils.getStackTrace(e);
				LOGGER.info("Thread interrupted.");
			}
		}
		long trainEnd = System.currentTimeMillis();

		if (performances.size() != this.experimentConfiguration.getNumberOfFolds()) {
			this.eventBus.post(new RegressorEvaluatedEvent(constructionInstruction, "", this.experimentConfiguration.getDatasetName(), exception, ""));
		} else {
			OptionalDouble pipelinePerformance = performances.stream().mapToDouble(x -> x).average();
			if (pipelinePerformance.isPresent()) {
				this.eventBus.post(new RegressorEvaluatedEvent(constructionInstruction, "", this.experimentConfiguration.getDatasetName(), pipelinePerformance.getAsDouble(), runtimes, ""));
				this.bestPipeline.update(new PipelineEvaluationReport(constructionInstruction, imports, pipelinePerformance.getAsDouble(), trainStart, trainEnd));
			}
		}
	}

	protected abstract AScikitLearnLearnerFactory getScikitLearnLearnerFactory();

	private ILabeledDataset<?> getTrainingDataset(final int split) {
		return this.datasetSplitSet.getFolds(split).get(0);
	}

	private ILabeledDataset<?> getTestingDataset(final int split) {
		return this.datasetSplitSet.getFolds(split).get(1);
	}

	protected abstract ScikitLearnWrapper<IPrediction, IPredictionBatch> setupScikitlearnWrapper(String constructionInstruction, String imports, final Timeout timeout) throws IOException;

	private IPredictionBatch runScikitLearnWrapper(final ScikitLearnWrapper<IPrediction, IPredictionBatch> scikitLearnWrapper, final int split, final Timeout timeout)
			throws AlgorithmTimeoutedException, ExecutionException, InterruptedException {
		return TimedComputation.compute(() -> scikitLearnWrapper.fitAndPredict(this.getTrainingDataset(split), this.getTestingDataset(split)), timeout, "Pipeline execution interrupted.");
	}

}
