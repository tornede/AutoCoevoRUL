package autocoevorul.regression;

import java.util.List;

import ai.libs.jaicore.components.api.IComponentInstance;
import autocoevorul.baseline.randomsearch.PipelineEvaluationReport;

public class RegressionGGPSolution extends PipelineEvaluationReport {

	private IComponentInstance regressorComponentInstance;
	private List<IComponentInstance> featureExtractorComponentInstances;

	public RegressionGGPSolution(final String constructionInstruction, final String imports, final IComponentInstance regressorComponentInstance,
			final List<IComponentInstance> featureExtractorComponentInstances, final Double performance) {
		super(constructionInstruction, imports, performance, 0, 0);
		this.regressorComponentInstance = regressorComponentInstance;
		this.featureExtractorComponentInstances = featureExtractorComponentInstances;
	}

	public IComponentInstance getRegressorComponentInstance() {
		return this.regressorComponentInstance;
	}

	public List<IComponentInstance> getFeatureExtractorComponentInstances() {
		return this.featureExtractorComponentInstances;
	}

}
