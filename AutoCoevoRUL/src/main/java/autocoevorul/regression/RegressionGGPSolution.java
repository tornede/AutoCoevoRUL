package autocoevorul.regression;

import ai.libs.jaicore.components.api.IComponentInstance;
import autocoevorul.baseline.randomsearch.PipelineEvaluationReport;

public class RegressionGGPSolution extends PipelineEvaluationReport {

	private IComponentInstance regressorComponentInstance;
	private IComponentInstance featureExtractorComponentInstances;

	public RegressionGGPSolution(final String constructionInstruction, final String imports, final IComponentInstance regressorComponentInstance,
			final IComponentInstance iComponentInstance, final Double performance) {
		super(constructionInstruction, imports, performance, 0, 0);
		this.regressorComponentInstance = regressorComponentInstance;
		this.featureExtractorComponentInstances = iComponentInstance;
	}

	public IComponentInstance getRegressorComponentInstance() {
		return this.regressorComponentInstance;
	}

	public IComponentInstance getFeatureExtractorComponentInstances() {
		return this.featureExtractorComponentInstances;
	}

}
