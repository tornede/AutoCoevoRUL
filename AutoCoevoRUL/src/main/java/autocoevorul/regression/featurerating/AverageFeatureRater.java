package autocoevorul.regression.featurerating;

import java.util.List;
import java.util.OptionalDouble;

import ai.libs.jaicore.ml.hpo.ggp.GrammarBasedGeneticProgramming.GGPSolutionCandidate;

public class AverageFeatureRater implements IFeatureRater {

	@Override
	public double rateFeatureExtractor(final List<GGPSolutionCandidate> solutionCandidatesWithFeatureExtractor) {
		OptionalDouble optionalResult = solutionCandidatesWithFeatureExtractor.stream().filter(candidate -> candidate.getScore() != null && !candidate.getScore().isNaN())
				.mapToDouble(candidate -> candidate.getScore()).average();
		if (optionalResult.isPresent()) {
			return optionalResult.getAsDouble();
		}
		return 10_000;
	}

}
