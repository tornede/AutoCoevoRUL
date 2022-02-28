package autocoevorul.regression.featurerating;

import java.util.List;

import ai.libs.jaicore.ml.hpo.ggp.GrammarBasedGeneticProgramming.GGPSolutionCandidate;

public class MedianFeatureRater implements IFeatureRater {

	@Override
	public double rateFeatureExtractor(final List<GGPSolutionCandidate> solutionCandidatesWithFeatureExtractor) {
		double[] optionalResult = solutionCandidatesWithFeatureExtractor.stream().filter(candidate -> candidate.getScore() != null && !candidate.getScore().isNaN()).mapToDouble(candidate -> candidate.getScore()).sorted().toArray();
		if (optionalResult != null && optionalResult.length > 0) {
			return optionalResult[(int) Math.floor(optionalResult.length / 2)];
		}
		return 10_000;
	}

}
