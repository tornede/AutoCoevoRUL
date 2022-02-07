package autocoevorul;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.StringJoiner;

import org.api4.java.ai.ml.core.evaluation.IPrediction;
import org.api4.java.ai.ml.core.evaluation.IPredictionBatch;
import org.junit.Test;
import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.BinaryIntegerVariable;
import org.moeaframework.core.variable.BinaryVariable;

import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnTimeSeriesFeatureEngineeringWrapper;
import ai.libs.mlplan.sklearn.ScikitLearnClassifierFactory;
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.featurerextraction.GenomeHandler;
import autocoevorul.featurerextraction.SolutionDecoding;

public class MoeaExampleProblemTest extends AbstractTest {

	@Test
	public void testCorrectDefinedSolution() throws Exception {
		ExperimentConfiguration experimentConfiguration = this.getExperimentConfiguration();
		GenomeHandler genomeHandler = this.setupGenomeHandler();

		Solution solution = this.getEmptySolution(experimentConfiguration, genomeHandler);
		((BinaryIntegerVariable) solution.getVariable(0)).setValue(this.getPositionInArray("True", "True", "False"));
		((BinaryVariable) solution.getVariable(65)).set(0, true);

		SolutionDecoding solutionDecoding = genomeHandler.decodeGenome(solution);
		List<IComponentInstance> componentInstances = solutionDecoding.getComponentInstances();
		componentInstances.sort(new Comparator<IComponentInstance>() {
			@Override
			public int compare(final IComponentInstance c1, final IComponentInstance c2) {
				return c1.getComponent().getName().compareTo(c2.getComponent().getName());
			}
		});

		assertEquals(1, componentInstances.size());

		this.executeScikitLearnWrapper(componentInstances);
	}

	private void executeScikitLearnWrapper(final List<IComponentInstance> componentInstances) throws Exception {
		ScikitLearnClassifierFactory factory = new ScikitLearnClassifierFactory();
		Set<String> importSet = new HashSet<>();

		StringJoiner constructionStringJoiner = new StringJoiner(",");
		for (IComponentInstance componentInstance : componentInstances) {
			constructionStringJoiner.add(factory.extractSKLearnConstructInstruction(componentInstance, importSet));
		}

		String constructionString = constructionStringJoiner.toString();
		if (componentInstances.size() > 1) {
			constructionString = "make_union(" + constructionString + ")";
		}

		StringBuilder imports = new StringBuilder();
		for (String importString : importSet) {
			imports.append(importString);
		}

		ScikitLearnTimeSeriesFeatureEngineeringWrapper<IPrediction, IPredictionBatch> sklearnWrapper = new ScikitLearnTimeSeriesFeatureEngineeringWrapper<>(constructionString, imports.toString());
		sklearnWrapper.setPythonTemplate(PYTHON_TEMPLATE_PATH);
		sklearnWrapper.fitAndPredict(this.getExperimentConfiguration().getTrainingData(), this.getExperimentConfiguration().getEvaluationData());
	}

}
