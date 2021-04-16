package autocoevorul.baseline.randomsearch;

import java.io.IOException;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.IPrediction;
import org.api4.java.ai.ml.core.evaluation.IPredictionBatch;
import org.api4.java.ai.ml.core.evaluation.execution.IDatasetSplitSet;
import org.api4.java.algorithm.Timeout;

import com.google.common.eventbus.EventBus;

import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.ml.core.EScikitLearnProblemType;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnWrapper;
import ai.libs.mlplan.sklearn.AScikitLearnLearnerFactory;
import ai.libs.mlplan.sklearn.ScikitLearnRULFactory;
import autocoevorul.SearchResult;
import autocoevorul.experiment.ExperimentConfiguration;

public class RulCompletePipelineEvaluator extends AbstractCompletePipelineEvaluator {

	public RulCompletePipelineEvaluator(final EventBus eventBus, final ExperimentConfiguration experimentConfiguration, final IDatasetSplitSet<ILabeledDataset<?>> datasetSplitSet,
			final Timeout timeout, final IComponentInstance componentInstance, final SearchResult bestPipeline) {
		super(eventBus, experimentConfiguration, datasetSplitSet, timeout, componentInstance, bestPipeline);
	}

	@Override
	protected ScikitLearnWrapper<IPrediction, IPredictionBatch> setupScikitlearnWrapper(final String constructionInstruction, final String imports, final Timeout timeout) throws IOException {
		ScikitLearnWrapper<IPrediction, IPredictionBatch> sklearnWrapper = new ScikitLearnWrapper<>(constructionInstruction, imports, false, EScikitLearnProblemType.RUL);
		sklearnWrapper.setScikitLearnWrapperConfig(this.experimentConfiguration.getScikitLearnWrapperConfig());
		sklearnWrapper.setTimeout(timeout);
		sklearnWrapper.setSeed(this.experimentConfiguration.getSeed());
		sklearnWrapper.setPythonTemplate(this.experimentConfiguration.getFeaturePythonTemplatePath());

		return sklearnWrapper;
	}

	@Override
	protected AScikitLearnLearnerFactory getScikitLearnLearnerFactory() {
		return new ScikitLearnRULFactory();
	}

}
