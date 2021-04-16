package autocoevorul.regression.featurerating;

import java.util.List;

import ai.libs.jaicore.ml.hpo.ggp.GrammarBasedGeneticProgramming.GGPSolutionCandidate;

public interface IFeatureRater {

	public double rateFeatureExtractor(List<GGPSolutionCandidate> solutionCandidatesWithFeatureExtractor);
}
