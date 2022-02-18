package autocoevorul.featurerextraction;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import org.api4.java.ai.ml.core.dataset.serialization.DatasetDeserializationFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.IPrediction;
import org.api4.java.ai.ml.core.evaluation.IPredictionBatch;
import org.api4.java.ai.ml.core.evaluation.execution.IDatasetSplitSet;
import org.api4.java.algorithm.exceptions.AlgorithmException;
import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.moeaframework.core.Solution;
import org.moeaframework.problem.AbstractProblem;
import org.moeaframework.problem.IBatchEvaluationProblem;
import org.slf4j.Logger;

import com.google.common.eventbus.EventBus;

import ai.libs.jaicore.components.exceptions.ComponentNotFoundException;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnTimeSeriesFeatureEngineeringWrapper;
import ai.libs.jaicore.timing.TimedComputation;
import autocoevorul.event.UpdatingBestPipelineEvent;
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.featurerextraction.genomehandler.GenomeHandler;
import autocoevorul.regression.RegressionGGPSolution;
import autocoevorul.regression.RegressionGgpProblem;

public class FeatureExtractionMoeaProblem extends AbstractProblem implements IBatchEvaluationProblem {

	private static final Logger LOGGER = org.slf4j.LoggerFactory.getLogger(FeatureExtractionMoeaProblem.class);

	private EventBus eventBus;

	private static final double WORST_OBJECTIVE_TIMEOUT = 10000.0;
	private static final double WORST_OBJECTIVE_INFEASIBLE = 20000.0;

	private final GenomeHandler genomeHandler;
	private final IDatasetSplitSet<ILabeledDataset<?>> datasetSplitSet;
	private ExperimentConfiguration experimentConfiguration;

	Map<Solution, SolutionDecoding> validSolutionDecodingsMap;
	private RegressionGGPSolution bestPipeline;

	public FeatureExtractionMoeaProblem(final EventBus eventBus, final ExperimentConfiguration experimentConfiguration, final GenomeHandler genomeHandler,
			final IDatasetSplitSet<ILabeledDataset<?>> datasetSplitSet) throws DatasetDeserializationFailedException {
		super(genomeHandler.getNumberOfVariables(), 2);
		this.eventBus = eventBus;
		this.genomeHandler = genomeHandler;
		this.datasetSplitSet = datasetSplitSet;
		this.experimentConfiguration = experimentConfiguration;
	}

	@Override
	public Solution newSolution() {
		return this.genomeHandler.newSolution();
	}

	private void setWorstObjeciveTimeout(final Solution solution) {
		solution.setObjective(0, WORST_OBJECTIVE_TIMEOUT);
		solution.setObjective(1, WORST_OBJECTIVE_TIMEOUT);
	}

	private void setWorstObjeciveInfeasible(final Solution solution) {
		solution.setObjective(0, WORST_OBJECTIVE_INFEASIBLE);
		solution.setObjective(1, WORST_OBJECTIVE_INFEASIBLE);
	}

	@Override
	public void evaluate(final Solution solution) {
		try {
			SolutionDecoding solutionDecoding = this.genomeHandler.decodeGenome(solution);
			if (solutionDecoding == null) {
				this.setWorstObjeciveInfeasible(solution);
			} else {
				try {
					for (int s = 0; s < this.datasetSplitSet.getNumberOfSplits(); s++) {
						if (Thread.interrupted()) {
							throw new InterruptedException();
						}
						ILabeledDataset<?> trainingData = this.datasetSplitSet.getFolds(s).get(0);
						ILabeledDataset<?> testingData = this.datasetSplitSet.getFolds(s).get(1);

						ScikitLearnTimeSeriesFeatureEngineeringWrapper<IPrediction, IPredictionBatch> sklearnWrapper = new ScikitLearnTimeSeriesFeatureEngineeringWrapper<>(solutionDecoding.getConstructionInstruction(), solutionDecoding.getImports());
						sklearnWrapper.setScikitLearnWrapperConfig(this.experimentConfiguration.getScikitLearnWrapperConfig());
						sklearnWrapper.setPythonTemplate(this.experimentConfiguration.getFeaturePythonTemplatePath());
						sklearnWrapper.setTimeout(this.experimentConfiguration.getFeatureCandidateTimeoutPerFold());
						sklearnWrapper.setSeed(this.experimentConfiguration.getSeed());

						TimedComputation.compute(() -> sklearnWrapper.fitAndPredict(trainingData, testingData), this.experimentConfiguration.getFeatureCandidateTimeoutPerFold(),
								"Feature engineering interrupted for fold " + s);
					}
					this.validSolutionDecodingsMap.put(solution, solutionDecoding);
				} catch (AlgorithmTimeoutedException | InterruptedException e) {
					this.setWorstObjeciveTimeout(solution);
				} catch (IOException | ExecutionException e) {
					this.setWorstObjeciveInfeasible(solution);
				}
			}
		} catch (ComponentNotFoundException e) {
			throw new RuntimeException("Decoding solution was not successful.", e);
		}
	}

	@Override
	public void evaluateAll(final List<Solution> batch) {
		this.validSolutionDecodingsMap = new ConcurrentHashMap<>(batch.size());
		ExecutorService pool = Executors.newFixedThreadPool(this.experimentConfiguration.getNumCPUs());
		for (Solution solution : batch) {
			pool.submit(new Runnable() {
				@Override
				public void run() {
					try {
						TimedComputation.compute(() -> {
							FeatureExtractionMoeaProblem.this.evaluate(solution);
							return true;
						}, FeatureExtractionMoeaProblem.this.experimentConfiguration.getFeatureCandidateTimeout(), "Feature engineering interrupted");
					} catch (AlgorithmTimeoutedException | InterruptedException e) {
						FeatureExtractionMoeaProblem.this.setWorstObjeciveTimeout(solution);
					} catch (ExecutionException e) {
						FeatureExtractionMoeaProblem.this.setWorstObjeciveInfeasible(solution);
					}
				}
			});
		}
		pool.shutdown();
		try {
			pool.awaitTermination(this.experimentConfiguration.getFeatureGenerationTimeout().milliseconds(), TimeUnit.MILLISECONDS);
		} catch (InterruptedException e) {
			LOGGER.error("Overall execution timeout exceeded for time series transformation. For all remaining solutions the objectives will be set to {}.", WORST_OBJECTIVE_TIMEOUT, e);
			for (Solution solution : batch) {
				if (!this.validSolutionDecodingsMap.containsKey(solution)) {
					this.setWorstObjeciveInfeasible(solution);
				}
			}
		}

		List<SolutionDecoding> validSolutionDecodingsList = this.validSolutionDecodingsMap.entrySet().stream().map(e -> e.getValue()).collect(Collectors.toList());
		LOGGER.info("Valid feature transformer found: {}", validSolutionDecodingsList.size());

		if (this.validSolutionDecodingsMap.size() > 0) {
			// Collect ground truth data
			List<List<Double>> groundTruthTest = new ArrayList<>();
			for (int fold = 0; fold < this.experimentConfiguration.getNumberOfFolds(); fold++) {
				groundTruthTest.add(this.datasetSplitSet.getFolds(fold).get(1).stream().map(instance -> (Double) instance.getLabel()).collect(Collectors.toList()));
			}

			try {
				RegressionGgpProblem runner = new RegressionGgpProblem(this.eventBus, this.experimentConfiguration, validSolutionDecodingsList, groundTruthTest);
				RegressionGGPSolution candidatePipeline = runner.evaluateExtractors();

				if (candidatePipeline != null) {
					if (this.bestPipeline == null || candidatePipeline.getPerformance() < this.bestPipeline.getPerformance()) {
						this.bestPipeline = candidatePipeline;
						LOGGER.info("Updating best solution found: {} {}", this.bestPipeline.getPerformance(), this.bestPipeline.getConstructionInstruction());

						UpdatingBestPipelineEvent updatingBestPipelineEvent = new UpdatingBestPipelineEvent(candidatePipeline.getConstructionInstruction(), candidatePipeline.getPerformance());
						this.eventBus.post(updatingBestPipelineEvent);

					}
				}
			} catch (IOException | AlgorithmTimeoutedException | AlgorithmExecutionCanceledException | AlgorithmException | InterruptedException e) {
				throw new RuntimeException(e);
			}
		}
	}

	public RegressionGGPSolution getBestPipeline() {
		return this.bestPipeline;
	}

}
