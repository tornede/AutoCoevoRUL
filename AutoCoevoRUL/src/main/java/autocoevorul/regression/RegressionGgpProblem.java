package autocoevorul.regression;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.api4.java.algorithm.exceptions.AlgorithmException;
import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.api4.java.common.attributedobjects.IObjectEvaluator;
import org.slf4j.Logger;

import com.google.common.eventbus.EventBus;

import ai.libs.jaicore.basic.ResourceFile;
import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.components.api.IComponentRepository;
import ai.libs.jaicore.components.model.Component;
import ai.libs.jaicore.components.model.NumericParameterDomain;
import ai.libs.jaicore.components.model.Parameter;
import ai.libs.jaicore.components.model.SoftwareConfigurationProblem;
import ai.libs.jaicore.components.serialization.ComponentSerialization;
import ai.libs.jaicore.ml.hpo.ggp.GrammarBasedGeneticProgramming;
import ai.libs.jaicore.ml.hpo.ggp.GrammarBasedGeneticProgramming.GGPSolutionCandidate;
import autocoevorul.event.FeatureExtractorEvaluatedEvent;
import autocoevorul.event.GGPRegressionResultFoundEvent;
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.featurerextraction.SolutionDecoding;
import autocoevorul.regression.featurerating.EFeatureRater;
import autocoevorul.util.ScikitLearnUtil;

// TODO THINK OF NAME
public class RegressionGgpProblem {

	private static final Logger LOGGER = org.slf4j.LoggerFactory.getLogger(RegressionGgpProblem.class);

	private EventBus eventBus;

	public static final String PLACEHOLDER_REQUIRED_INTERFACE_TO_SEARCH = "ComponentToSearch";
	public static final String PLACEHOLDER_FEATURE_EXTRACTOR_ID_PARAMETER_NAME = "feature_extractor_id";

	private ExperimentConfiguration experimentConfiguration;
	private List<SolutionDecoding> featureDecodings;
	private final List<List<Double>> groundTruthsForSplits;
	private SoftwareConfigurationProblem<Double> softwareConfigurationProblem;

	private GGPSolutionCandidate ggpResult;

	public RegressionGgpProblem(final EventBus eventBus, final ExperimentConfiguration experimentConfiguration, final List<SolutionDecoding> featureDecodings, final List<List<Double>> groundTruthTest) throws IOException {
		if (featureDecodings.size() == 0) {
			LOGGER.error("Given list of feature decodings is empty, which is not permitted.", featureDecodings.size());
			throw new RuntimeException("Given list of feature decodings is empty, which is not permitted.");
		}
		this.eventBus = eventBus;
		this.experimentConfiguration = experimentConfiguration;
		this.featureDecodings = featureDecodings;
		this.groundTruthsForSplits = groundTruthTest;
		this.softwareConfigurationProblem = this.constructSoftwareConfigurationProblem();
	}

	private SoftwareConfigurationProblem<Double> constructSoftwareConfigurationProblem() throws IOException {
		IComponentRepository componentRepository = new ComponentSerialization().deserializeRepository(new ResourceFile(this.experimentConfiguration.getRegressionSearchpace()));

		// create dummy component featuring a parameter for the feature extractor which we are actually using
		Component componentToCompose = new Component(PLACEHOLDER_REQUIRED_INTERFACE_TO_SEARCH);
		componentToCompose.addProvidedInterface(PLACEHOLDER_REQUIRED_INTERFACE_TO_SEARCH);
		componentToCompose.addParameter(new Parameter(PLACEHOLDER_FEATURE_EXTRACTOR_ID_PARAMETER_NAME, new NumericParameterDomain(true, 0, this.featureDecodings.size() - 1), 0));
		componentToCompose.addRequiredInterface(this.experimentConfiguration.getRegressionRequiredInterface(), this.experimentConfiguration.getRegressionRequiredInterface());

		// add this dummy component to the repository
		componentRepository.add(componentToCompose);

		IObjectEvaluator<IComponentInstance, Double> regressionEvaluator = new CompletePipelineEvaluator(this.eventBus, this.experimentConfiguration, this.featureDecodings, this.groundTruthsForSplits);

		return new SoftwareConfigurationProblem<>(componentRepository, PLACEHOLDER_REQUIRED_INTERFACE_TO_SEARCH, regressionEvaluator);
	}

	public RegressionGGPSolution evaluateExtractors() throws AlgorithmTimeoutedException, InterruptedException, AlgorithmExecutionCanceledException, AlgorithmException {
		List<GGPSolutionCandidate> solutionCandidatesFromLastPopulation = this.runOptimizationOfRegressorsUsingExtractors();
		Map<String, List<GGPSolutionCandidate>> featureExtractorStringsToSolutionCandidatesWithExtractor = this.mapGGPSolutionCandidatesToFeatureExtractor(solutionCandidatesFromLastPopulation);

		for (SolutionDecoding featureDecoding : this.featureDecodings) {
			List<GGPSolutionCandidate> solutionCandidatesWithFeatureExtractor = featureExtractorStringsToSolutionCandidatesWithExtractor.get(featureDecoding.getConstructionInstruction());
			double featureRating = this.experimentConfiguration.getFeatureObjectiveMeasure().rateFeatureExtractor(solutionCandidatesWithFeatureExtractor);
			double numberOfUsageRating = EFeatureRater.NUMBER_OF_USING_REGRESSORS.rateFeatureExtractor(solutionCandidatesWithFeatureExtractor);
			featureDecoding.setPerformance(featureRating, numberOfUsageRating);

			List<Double> performancesOfIncludingRegressors = featureExtractorStringsToSolutionCandidatesWithExtractor.get(featureDecoding.getConstructionInstruction()).stream().map(g -> g.getScore()).collect(Collectors.toList());
			FeatureExtractorEvaluatedEvent event = new FeatureExtractorEvaluatedEvent(featureDecoding.getConstructionInstruction(), this.experimentConfiguration.getDatasetName(), featureRating, performancesOfIncludingRegressors,
					numberOfUsageRating);
			this.eventBus.post(event);
		}

		return this.createRegressionGGPSolutionFromGGPResult();
	}

	private RegressionGGPSolution createRegressionGGPSolutionFromGGPResult() {
		if (this.ggpResult == null) {
			RegressionGGPSolution regressionGGPSolution = new RegressionGGPSolution("None", "None", null, null, 10_000.0);
			this.eventBus.post(new GGPRegressionResultFoundEvent(regressionGGPSolution));
			return null;
		}
		Pair<String, String> featureTransformerConstructionInstructionAndImports = this.getFeatureTransformerConstructionInstructionFromGGPResult();
		Pair<String, String> regressorConstructionInstructionAndImports = this.getRegressorConstructionInstructionFromGGPResult();

		String constructionString = "make_pipeline(" + featureTransformerConstructionInstructionAndImports.getX() + "," + regressorConstructionInstructionAndImports.getX() + ")";
		String imports = featureTransformerConstructionInstructionAndImports.getY() + "\n" + regressorConstructionInstructionAndImports.getY();

		RegressionGGPSolution regressionGGPSolution = new RegressionGGPSolution(constructionString, imports, this.getRegressionComponentInstanceFromGGPResult(), this.getSolutionDecodingOfExtractorPresentInGGPResult().getComponentInstance(),
				this.ggpResult.getScore());

		this.eventBus.post(new GGPRegressionResultFoundEvent(regressionGGPSolution));

		return regressionGGPSolution;
	}

	private Pair<String, String> getFeatureTransformerConstructionInstructionFromGGPResult() {
		SolutionDecoding featureDecodingOfExtractorPresentInResult = this.getSolutionDecodingOfExtractorPresentInGGPResult();
		String featureTransformerConstructionInstruction = featureDecodingOfExtractorPresentInResult.getConstructionInstruction();

		return new Pair<>(featureTransformerConstructionInstruction, featureDecodingOfExtractorPresentInResult.getImports());
	}

	private SolutionDecoding getSolutionDecodingOfExtractorPresentInGGPResult() {
		int featureExtractorId = Integer.parseInt(this.ggpResult.getComponentInstance().getParameterValue(PLACEHOLDER_FEATURE_EXTRACTOR_ID_PARAMETER_NAME));
		SolutionDecoding featureDecodingOfExtractorPresentInResult = this.featureDecodings.get(featureExtractorId);
		return featureDecodingOfExtractorPresentInResult;
	}

	private Pair<String, String> getRegressorConstructionInstructionFromGGPResult() {
		IComponentInstance regressorComponentInstance = this.getRegressionComponentInstanceFromGGPResult();
		return ScikitLearnUtil.createConstructionInstructionAndImportsFromComponentInstance(regressorComponentInstance);
	}

	private IComponentInstance getRegressionComponentInstanceFromGGPResult() {
		IComponentInstance regressorComponentInstance = this.ggpResult.getComponentInstance();
		regressorComponentInstance = regressorComponentInstance.getSatisfactionOfRequiredInterface(this.experimentConfiguration.getRegressionRequiredInterface()).get(0);
		return regressorComponentInstance;
	}

	private List<GGPSolutionCandidate> runOptimizationOfRegressorsUsingExtractors() throws AlgorithmTimeoutedException, InterruptedException, AlgorithmExecutionCanceledException, AlgorithmException {
		GrammarBasedGeneticProgramming ggp = new GrammarBasedGeneticProgramming(this.experimentConfiguration.getRegressionGGPConfig(), this.softwareConfigurationProblem, this.experimentConfiguration.getSeed());
		this.ggpResult = ggp.call();
		if (this.ggpResult != null) {
			LOGGER.info("FOUND BEST SOLUTION: {} {}", this.ggpResult.getScore(), this.ggpResult.getComponentInstance());
		}
		return ggp.getLastRatedPopulation();
	}

	private Map<String, List<GGPSolutionCandidate>> mapGGPSolutionCandidatesToFeatureExtractor(final List<GGPSolutionCandidate> solutionCandidates) {
		Map<String, List<GGPSolutionCandidate>> featureExtractorToIncludingSolutionCandidates = new HashMap<>();
		for (SolutionDecoding featureExtractorDecoding : this.featureDecodings) {
			featureExtractorToIncludingSolutionCandidates.put(featureExtractorDecoding.getConstructionInstruction(), new ArrayList<>());
		}

		for (GGPSolutionCandidate solutionCandidate : solutionCandidates) {
			int featureExtractorId = Integer.parseInt(solutionCandidate.getComponentInstance().getParameterValue(PLACEHOLDER_FEATURE_EXTRACTOR_ID_PARAMETER_NAME));
			String featureExtractorString = this.featureDecodings.get(featureExtractorId).getConstructionInstruction();
			featureExtractorToIncludingSolutionCandidates.get(featureExtractorString).add(solutionCandidate);
		}
		return featureExtractorToIncludingSolutionCandidates;
	}

}
