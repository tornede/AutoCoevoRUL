package autocoevorul.regression.featurerating;

import java.util.List;

import ai.libs.jaicore.ml.hpo.ggp.GrammarBasedGeneticProgramming.GGPSolutionCandidate;

public class NumberOfUsingRegressorsFeatureRater implements IFeatureRater {

	@Override
	public double rateFeatureExtractor(final List<GGPSolutionCandidate> solutionCandidatesWithFeatureExtractor) {
		// Return negative value as we assume low values to be very good
		return -solutionCandidatesWithFeatureExtractor.size();
	}

}
