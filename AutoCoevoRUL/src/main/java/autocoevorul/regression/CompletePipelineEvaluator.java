package autocoevorul.regression;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.lang3.StringUtils;
import org.api4.java.ai.ml.core.evaluation.IPrediction;
import org.api4.java.ai.ml.core.evaluation.IPredictionBatch;
import org.api4.java.ai.ml.core.evaluation.execution.IAggregatedPredictionPerformanceMeasure;
import org.api4.java.ai.ml.core.evaluation.execution.ILearnerRunReport;
import org.api4.java.ai.ml.regression.evaluation.IRegressionPrediction;
import org.api4.java.algorithm.Timeout;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.api4.java.common.attributedobjects.IObjectEvaluator;
import org.api4.java.common.attributedobjects.ObjectEvaluationFailedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.eventbus.EventBus;
import com.google.common.hash.Hashing;

import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.ml.core.evaluation.AveragingPredictionPerformanceMeasure;
import ai.libs.jaicore.ml.core.evaluation.evaluator.LearnerRunReport;
import ai.libs.jaicore.ml.core.evaluation.evaluator.TypelessPredictionDiff;
import ai.libs.jaicore.ml.regression.singlelabel.SingleTargetRegressionPrediction;
import ai.libs.jaicore.ml.scikitwrapper.AScikitLearnWrapper;
import ai.libs.jaicore.ml.scikitwrapper.IScikitLearnWrapperConfig;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnRegressionWrapper;
import ai.libs.jaicore.search.algorithms.standard.bestfirst.nodeevaluation.TimeAwareNodeEvaluator;
import ai.libs.jaicore.timing.TimedComputation;
import autocoevorul.event.RegressorEvaluatedEvent;
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.featurerextraction.SolutionDecoding;
import autocoevorul.util.ScikitLearnUtil;

public class CompletePipelineEvaluator implements IObjectEvaluator<IComponentInstance, Double> {

	private Logger logger = LoggerFactory.getLogger(CompletePipelineEvaluator.class);

	private EventBus eventBus;

	private final ExperimentConfiguration experimentConfiguration;
	private final List<SolutionDecoding> solutionDecodings;
	private final List<List<? extends Double>> groundTruthsForSplits;
	private final IAggregatedPredictionPerformanceMeasure<Double, IRegressionPrediction> metric;

	public CompletePipelineEvaluator(final EventBus eventBus, final ExperimentConfiguration experimentConfiguration, final List<SolutionDecoding> featureExtractorStrings,
			final List<List<Double>> groundTruthsForSplits) {
		super();
		this.eventBus = eventBus;
		this.experimentConfiguration = experimentConfiguration;
		this.solutionDecodings = featureExtractorStrings;

		this.groundTruthsForSplits = new ArrayList<>(experimentConfiguration.getNumberOfFolds());
		for (List<Double> groundTruthsForSplit : groundTruthsForSplits) {
			this.groundTruthsForSplits.add(new ArrayList<>(groundTruthsForSplit));
		}

		this.metric = new AveragingPredictionPerformanceMeasure<Double, IRegressionPrediction>(experimentConfiguration.getPerformanceMeasure());
	}

	@Override
	public Double evaluate(final IComponentInstance componentInstance) throws InterruptedException, ObjectEvaluationFailedException {

		// extract top element from component instance as this only holds the dummy
		// parameter describing the dataset to use
		int idOfFeatureExtractorToUse = Integer.parseInt(componentInstance.getParameterValue(RegressionGgpProblem.PLACEHOLDER_FEATURE_EXTRACTOR_ID_PARAMETER_NAME));

		String featureExtractorConstructionString = this.solutionDecodings.get(idOfFeatureExtractorToUse).getConstructionInstruction();
		String hashCode = Hashing.sha256().hashString(StringUtils.join(featureExtractorConstructionString, this.solutionDecodings.get(idOfFeatureExtractorToUse).getImports()), StandardCharsets.UTF_8)
				.toString();
		String featureExtractorHashcode = hashCode.startsWith("-") ? hashCode.replace("-", "1") : "0" + hashCode;

		List<ILearnerRunReport> reports = new ArrayList<>(this.experimentConfiguration.getNumberOfFolds());
		List<List<? extends IRegressionPrediction>> predictions = new ArrayList<>(this.experimentConfiguration.getNumberOfFolds());
		List<Long> runtimes = new ArrayList<>(this.experimentConfiguration.getNumberOfFolds());
		String pipeline = "";
		String exception = "";

		long trainStart = 0;
		long trainEnd = 0;
		for (int i = 0; i < this.experimentConfiguration.getNumberOfFolds(); i++) {
			if (Thread.interrupted()) {
				throw new InterruptedException();
			}

			String trainDatasetName = this.getTrainingDatasetName(featureExtractorHashcode, i);
			String testDatasetName = this.getTestDatasetName(featureExtractorHashcode, i);

			ILearnerRunReport report = null;
			ScikitLearnRegressionWrapper<IPrediction, IPredictionBatch> learner = null;
			try {

				learner = this.createScikitlearnWrapper(componentInstance);
				
				File testDatasetFile = this.getTrainingDatasetFile(learner, testDatasetName); // TODO
				File trainDatasetFile = this.getTrainingDatasetFile(learner, trainDatasetName); // TODO
				
				pipeline = learner.toString();

				trainStart = System.currentTimeMillis();
				// IPredictionBatch predictionsForSplit = learner.fitAndPredict(trainDatasetName, testDatasetName);
				IPredictionBatch predictionsForSplit = this.fitAndPredictWithWrapperUnderTimeout(learner, trainDatasetFile, trainDatasetName, testDatasetFile, testDatasetName);
				trainEnd = System.currentTimeMillis();

				runtimes.add(trainEnd - trainStart);

				List<IRegressionPrediction> doublePredictionsForSplit = predictionsForSplit.getPredictions().stream()
						.map(prediction -> new SingleTargetRegressionPrediction(Math.max(0, ((IRegressionPrediction) prediction).getDoublePrediction()))).collect(Collectors.toList());

				report = new LearnerRunReport(null, null, 0, 0, trainStart, trainEnd, new TypelessPredictionDiff(this.groundTruthsForSplits.get(i), doublePredictionsForSplit));

				predictions.add(doublePredictionsForSplit);

			} catch (IOException e) {
				exception = ExceptionUtils.getStackTrace(e);
				this.logger.warn("IOException during training of Scikitlearn wrapper with pipeline {} on dataset {}. Error: \n {}", learner, this.experimentConfiguration.getDatasetName(), exception);
				reports.add(new LearnerRunReport(null, null, trainStart, System.currentTimeMillis(), e));
				break;
			} catch (AlgorithmTimeoutedException e) {
				exception = ExceptionUtils.getStackTrace(e);
				this.logger.warn("Failed to fit and predict with Scikitlearn wrapper with pipeline {} on dataset {} due to a timeout.", learner, this.experimentConfiguration.getDatasetName());
				reports.add(new LearnerRunReport(null, null, trainStart, System.currentTimeMillis(), e));
				break;
			} catch (ExecutionException e) {
				exception = ExceptionUtils.getStackTrace(e);
				this.logger.warn("Failed to fit and predict with Scikitlearn wrapper with pipeline {} on dataset {}. Error: \n {}", learner, this.experimentConfiguration.getDatasetName(), exception);
				reports.add(new LearnerRunReport(null, null, trainStart, System.currentTimeMillis(), e));
				break;
			}
			if (report != null) {
				reports.add(report);
			}
		}
		if (predictions.size() < this.groundTruthsForSplits.size()) {
			// all folds have failed, so we have to throw an ObjectEvaluationFailedException
			this.eventBus.post(new RegressorEvaluatedEvent(pipeline, featureExtractorConstructionString, this.getTrainingDatasetName(featureExtractorHashcode, -1), exception,
					componentInstance.getAnnotation("generation")));
			throw new ObjectEvaluationFailedException("Could not evaluate learner " + pipeline + " as at least one fold has failed.");
		}

		this.logger.debug("Compute metric ({}) for the diff of predictions and ground truth.", this.metric.getClass().getName());
		double score = this.metric.loss(this.groundTruthsForSplits, predictions);
		this.logger.info("Computed value for metric {} of {} executions. Metric value is: {}. Pipeline: {}", this.metric, this.experimentConfiguration.getNumberOfFolds(), score, pipeline);
		this.eventBus.post(new RegressorEvaluatedEvent(pipeline, featureExtractorConstructionString, this.getTrainingDatasetName(featureExtractorHashcode, -1), score, runtimes,
				componentInstance.getAnnotation("generation")));

		return score;
	}

	private String getTestDatasetName(final String featureExtractorHashcode, final int fold) {
		return featureExtractorHashcode + "_" + this.experimentConfiguration.getDatasetName() + "_" + this.experimentConfiguration.getSeed() + "_" + fold + "_test";
	}

	private String getTrainingDatasetName(final String featureExtractorHashcode, final int fold) {
		return featureExtractorHashcode + "_" + this.experimentConfiguration.getDatasetName() + "_" + this.experimentConfiguration.getSeed() + "_" + fold + "_train";
	}
	
	private File getTrainingDatasetFile(ScikitLearnRegressionWrapper<IPrediction, IPredictionBatch> learner, final String datasetName) {
		try {
			Field f = AScikitLearnWrapper.class.getDeclaredField("scikitLearnWrapperConfig");
			f.setAccessible(true);
			
			Method x = IScikitLearnWrapperConfig.class.getDeclaredMethod("getTempFolder");
			x.setAccessible(true);
			
			File tmpFolder = (File) x.invoke(f.get(learner));
			return new File(tmpFolder + "/" + datasetName);
			
		} catch (IllegalAccessException | IllegalArgumentException | InvocationTargetException | NoSuchFieldException | NoSuchMethodException | SecurityException e) {
			throw new RuntimeException("Could not figure out correct file for training or test dataset: " + datasetName);
		}
		
	}

	public IPredictionBatch fitAndPredictWithWrapperUnderTimeout(final ScikitLearnRegressionWrapper<IPrediction, IPredictionBatch> wrapper, final File trainDatasetFile, final String trainDatasetName, final File testDatasetFile, final String testDatasetName)
			throws AlgorithmTimeoutedException, ExecutionException, InterruptedException {
		return TimedComputation.compute(() -> wrapper.fitAndPredict(trainDatasetFile, trainDatasetName, testDatasetFile, testDatasetName),
				new Timeout(this.experimentConfiguration.getRegressionCandidateTimeout().seconds() + 2, TimeUnit.SECONDS),
				"Node evaluation has timed out (" + TimeAwareNodeEvaluator.class.getName() + "::" + Thread.currentThread() + "-" + System.currentTimeMillis() + ")");
	}

	private ScikitLearnRegressionWrapper<IPrediction, IPredictionBatch> createScikitlearnWrapper(final IComponentInstance componentInstance) throws IOException, InterruptedException {
		List<IComponentInstance> satisfiedRegressorInterfaceInstancesInComponentInstance = componentInstance
				.getSatisfactionOfRequiredInterface(this.experimentConfiguration.getRegressionRequiredInterface());

		if (satisfiedRegressorInterfaceInstancesInComponentInstance.size() != 1) {
			throw new RuntimeException("More or less than one regressor under dummy component!");
		}
		IComponentInstance componentInstanceWithoutDummy = satisfiedRegressorInterfaceInstancesInComponentInstance.get(0);

		Pair<String, String> constructionInstructionAndImports = ScikitLearnUtil.createConstructionInstructionAndImportsFromComponentInstance(componentInstanceWithoutDummy);

		ScikitLearnRegressionWrapper<IPrediction, IPredictionBatch> sklearnWrapper = new ScikitLearnRegressionWrapper<>(constructionInstructionAndImports.getX(), constructionInstructionAndImports.getY());
		sklearnWrapper.setScikitLearnWrapperConfig(this.experimentConfiguration.getScikitLearnWrapperConfig());
		sklearnWrapper.setTimeout(this.experimentConfiguration.getRegressionCandidateTimeout());
		sklearnWrapper.setSeed(this.experimentConfiguration.getSeed());
		return sklearnWrapper;
	}

}
