package autocoevorul.baseline.randomsearch;

import java.io.IOException;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.execution.IDatasetSplitSet;
import org.api4.java.algorithm.Timeout;

import com.google.common.eventbus.EventBus;

import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnTimeSeriesRegressionWrapper;
import autocoevorul.SearchResult;
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.featurerextraction.ML4PdMFactory;

public class RulCompletePipelineEvaluator extends AbstractCompletePipelineEvaluator {

	public RulCompletePipelineEvaluator(final EventBus eventBus, final ExperimentConfiguration experimentConfiguration, final IDatasetSplitSet<ILabeledDataset<?>> datasetSplitSet, final Timeout timeout,
			final IComponentInstance componentInstance, final SearchResult bestPipeline) {
		super(eventBus, experimentConfiguration, datasetSplitSet, timeout, componentInstance, bestPipeline);
	}

	@Override
	protected ScikitLearnTimeSeriesRegressionWrapper setupScikitlearnWrapper(final String constructionInstruction, final String imports, final Timeout timeout) throws IOException, InterruptedException {
		ScikitLearnTimeSeriesRegressionWrapper sklearnWrapper = new ScikitLearnTimeSeriesRegressionWrapper(constructionInstruction, imports);
		sklearnWrapper.setScikitLearnWrapperConfig(this.experimentConfiguration.getScikitLearnWrapperConfig());
		sklearnWrapper.setTimeout(timeout);
		sklearnWrapper.setSeed(this.experimentConfiguration.getSeed());
		return sklearnWrapper;
	}

	@Override
	protected ML4PdMFactory getScikitLearnLearnerFactory() {
		return new ML4PdMFactory();
	}

}
