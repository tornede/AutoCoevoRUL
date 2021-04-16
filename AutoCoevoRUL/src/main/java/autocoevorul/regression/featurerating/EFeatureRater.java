package autocoevorul.regression.featurerating;

import java.util.List;

import ai.libs.jaicore.ml.hpo.ggp.GrammarBasedGeneticProgramming.GGPSolutionCandidate;

public enum EFeatureRater implements IFeatureRater {
	AVERAGE(new AverageFeatureRater()), //
	NUMBER_OF_USING_REGRESSORS(new NumberOfUsingRegressorsFeatureRater()), //
	MINIMUM(new MinimumFeatureRater()), //
	MEDIAN(new MedianFeatureRater());

	private final IFeatureRater measure;

	private EFeatureRater(final IFeatureRater measure) {
		this.measure = measure;
	}

	@Override
	public double rateFeatureExtractor(final List<GGPSolutionCandidate> solutionCandidatesWithFeatureExtractor) {
		return this.measure.rateFeatureExtractor(solutionCandidatesWithFeatureExtractor);
	}

}
