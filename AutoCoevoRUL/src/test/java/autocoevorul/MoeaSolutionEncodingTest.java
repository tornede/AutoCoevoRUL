package autocoevorul;

import static org.junit.Assert.assertNull;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.util.Comparator;
import java.util.List;

import org.api4.java.ai.ml.core.dataset.serialization.DatasetDeserializationFailedException;
import org.junit.Test;
import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.BinaryIntegerVariable;
import org.moeaframework.core.variable.BinaryVariable;

import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.components.api.IParameter;
import ai.libs.jaicore.components.exceptions.ComponentNotFoundException;
import autocoevorul.featurerextraction.GenomeHandler;
import autocoevorul.featurerextraction.SolutionDecoding;

public class MoeaSolutionEncodingTest extends AbstractTest {

	@Test
	public void testCorrectDefinedSolution() throws IOException, ComponentNotFoundException, DatasetDeserializationFailedException {
		GenomeHandler genomeHandler = new GenomeHandler(this.getExperimentConfiguration());

		Solution solution = this.getEmptySolution(this.getExperimentConfiguration(), genomeHandler);
		((BinaryVariable) solution.getVariable(65)).set(0, true);
		((BinaryVariable) solution.getVariable(68)).set(0, true);
		((BinaryIntegerVariable) solution.getVariable(69)).setValue(this.getPositionInArray("True", "True", "False"));
		((BinaryIntegerVariable) solution.getVariable(70)).setValue(this.getPositionInArray("mutual_info", "mutual_info", "anova"));
		((BinaryIntegerVariable) solution.getVariable(71)).setValue(this.getPositionInArray("False", "True", "False"));
		((BinaryIntegerVariable) solution.getVariable(72)).setValue(5);

		SolutionDecoding solutionDecoding = genomeHandler.decodeGenome(solution);
		List<IComponentInstance> componentInstances = solutionDecoding.getComponentInstances();
		assertEquals(1, componentInstances.size());

		IComponentInstance rootComponentInstance = componentInstances.get(0);
		assertEquals("python_connection.feature_generation.uni_to_multi_numpy_feature_generator.UniToMultivariateNumpyBasedFeatureGenerator", rootComponentInstance.getComponent().getName());

		IComponentInstance maskingComponentInstance = rootComponentInstance.getSatisfactionOfRequiredInterface("masking_strategy").get(0);
		assertEquals("python_connection.feature_generation.uni_to_multi_numpy_feature_generator.RandomValueMaskingStrategy", maskingComponentInstance.getComponent().getName());

		IComponentInstance fgComponentInstance = rootComponentInstance.getSatisfactionOfRequiredInterface("univariate_ts_feature_generator").get(0);
		assertEquals("pyts.transformation.ShapeletTransform", fgComponentInstance.getComponent().getName());
		assertEquals("True", fgComponentInstance.getParameterValue("remove_similar"));
		assertEquals("mutual_info", fgComponentInstance.getParameterValue("criterion"));
		assertEquals("False", fgComponentInstance.getParameterValue("sort"));
	}

	@Test
	public void testInactiveDefinedSolution() throws IOException, ComponentNotFoundException, DatasetDeserializationFailedException {
		GenomeHandler genomeHandler = new GenomeHandler(this.getExperimentConfiguration());

		Solution solution = this.getEmptySolution(this.getExperimentConfiguration(), genomeHandler);
		SolutionDecoding solutionDecoding = genomeHandler.decodeGenome(solution);
		assertNull(solutionDecoding);
	}

	@Test
	public void testIncompleteDefinedSolution() throws IOException, ComponentNotFoundException, DatasetDeserializationFailedException {
		GenomeHandler genomeHandler = new GenomeHandler(this.getExperimentConfiguration());

		Solution solution = this.getEmptySolution(this.getExperimentConfiguration(), genomeHandler);
		((BinaryVariable) solution.getVariable(68)).set(0, true);
		((BinaryIntegerVariable) solution.getVariable(69)).setValue(this.getPositionInArray("True", "True", "False"));
		((BinaryIntegerVariable) solution.getVariable(70)).setValue(this.getPositionInArray("mutual_info", "mutual_info", "anova"));
		((BinaryIntegerVariable) solution.getVariable(71)).setValue(this.getPositionInArray("False", "True", "False"));
		((BinaryIntegerVariable) solution.getVariable(72)).setValue(5);

		SolutionDecoding solutionDecoding = genomeHandler.decodeGenome(solution);
		assertNull(solutionDecoding);
	}

	@Test
	public void testCorrectDefinedSolutionWithoutTsfresh() throws IOException, ComponentNotFoundException, DatasetDeserializationFailedException {
		GenomeHandler genomeHandler = new GenomeHandler(this.getExperimentConfiguration());

		Solution solution = this.getEmptySolution(this.getExperimentConfiguration(), genomeHandler);
		((BinaryVariable) solution.getVariable(65)).set(0, true);
		((BinaryVariable) solution.getVariable(68)).set(0, true);
		((BinaryIntegerVariable) solution.getVariable(69)).setValue(this.getPositionInArray("True", "True", "False"));
		((BinaryIntegerVariable) solution.getVariable(70)).setValue(this.getPositionInArray("mutual_info", "mutual_info", "anova"));
		((BinaryIntegerVariable) solution.getVariable(71)).setValue(this.getPositionInArray("False", "True", "False"));
		((BinaryIntegerVariable) solution.getVariable(72)).setValue(5);

		SolutionDecoding solutionDecoding = genomeHandler.decodeGenome(solution);
		List<IComponentInstance> componentInstances = solutionDecoding.getComponentInstances();
		assertEquals(1, componentInstances.size());

		IComponentInstance rootComponentInstance = componentInstances.get(0);
		assertEquals("python_connection.feature_generation.uni_to_multi_numpy_feature_generator.UniToMultivariateNumpyBasedFeatureGenerator", rootComponentInstance.getComponent().getName());

		IComponentInstance maskingComponentInstance = rootComponentInstance.getSatisfactionOfRequiredInterface("masking_strategy").get(0);
		assertEquals("python_connection.feature_generation.uni_to_multi_numpy_feature_generator.RandomValueMaskingStrategy", maskingComponentInstance.getComponent().getName());

		IComponentInstance fgComponentInstance = rootComponentInstance.getSatisfactionOfRequiredInterface("univariate_ts_feature_generator").get(0);
		assertEquals("pyts.transformation.ShapeletTransform", fgComponentInstance.getComponent().getName());
		assertEquals("True", fgComponentInstance.getParameterValue("remove_similar"));
		assertEquals("mutual_info", fgComponentInstance.getParameterValue("criterion"));
		assertEquals("False", fgComponentInstance.getParameterValue("sort"));
	}

	@Test
	public void testCorrectDefinedSolutionWithTsfresh() throws IOException, ComponentNotFoundException, DatasetDeserializationFailedException {
		GenomeHandler genomeHandler = new GenomeHandler(this.getExperimentConfiguration());

		Solution solution = this.getEmptySolution(this.getExperimentConfiguration(), genomeHandler);
		((BinaryIntegerVariable) solution.getVariable(0)).setValue(this.getPositionInArray("True", "True", "False"));
		((BinaryVariable) solution.getVariable(65)).set(0, true);
		((BinaryVariable) solution.getVariable(68)).set(0, true);
		((BinaryIntegerVariable) solution.getVariable(69)).setValue(this.getPositionInArray("True", "True", "False"));
		((BinaryIntegerVariable) solution.getVariable(70)).setValue(this.getPositionInArray("mutual_info", "mutual_info", "anova"));
		((BinaryIntegerVariable) solution.getVariable(71)).setValue(this.getPositionInArray("False", "True", "False"));
		((BinaryIntegerVariable) solution.getVariable(72)).setValue(5);

		SolutionDecoding solutionDecoding = genomeHandler.decodeGenome(solution);
		List<IComponentInstance> componentInstances = solutionDecoding.getComponentInstances();
		componentInstances.sort(new Comparator<IComponentInstance>() {
			@Override
			public int compare(final IComponentInstance c1, final IComponentInstance c2) {
				return c1.getComponent().getName().compareTo(c2.getComponent().getName());
			}
		});

		assertEquals(2, componentInstances.size());

		IComponentInstance rootComponentInstance = componentInstances.get(0);
		assertEquals("python_connection.feature_generation.tsfresh_feature_generator.TsFreshBasedFeatureExtractor", rootComponentInstance.getComponent().getName());

		IComponentInstance dictionaryComponentInstance = rootComponentInstance.getSatisfactionOfRequiredInterface("default_fc_parameters").get(0);
		assertEquals("python_connection.feature_generation.tsfresh_feature_generator.FCParametersDictionary", dictionaryComponentInstance.getComponent().getName());

		for (IParameter parameter : dictionaryComponentInstance.getComponent().getParameters()) {
			if (parameter.getName().equals("minimum")) {
				assertEquals("True", dictionaryComponentInstance.getParameterValue(parameter.getName()));
			} else {
				assertEquals("False", dictionaryComponentInstance.getParameterValue(parameter.getName()));
			}
		}

		rootComponentInstance = componentInstances.get(1);
		assertEquals("python_connection.feature_generation.uni_to_multi_numpy_feature_generator.UniToMultivariateNumpyBasedFeatureGenerator", rootComponentInstance.getComponent().getName());

		IComponentInstance maskingComponentInstance = rootComponentInstance.getSatisfactionOfRequiredInterface("masking_strategy").get(0);
		assertEquals("python_connection.feature_generation.uni_to_multi_numpy_feature_generator.RandomValueMaskingStrategy", maskingComponentInstance.getComponent().getName());

		IComponentInstance fgComponentInstance = rootComponentInstance.getSatisfactionOfRequiredInterface("univariate_ts_feature_generator").get(0);
		assertEquals("pyts.transformation.ShapeletTransform", fgComponentInstance.getComponent().getName());
		assertEquals("True", fgComponentInstance.getParameterValue("remove_similar"));
		assertEquals("mutual_info", fgComponentInstance.getParameterValue("criterion"));
		assertEquals("False", fgComponentInstance.getParameterValue("sort"));
	}

	@Test
	public void testCorrectDefinedSolutionWithOnlyTsfresh() throws IOException, ComponentNotFoundException, DatasetDeserializationFailedException {
		GenomeHandler genomeHandler = new GenomeHandler(this.getExperimentConfiguration());

		Solution solution = this.getEmptySolution(this.getExperimentConfiguration(), genomeHandler);
		((BinaryIntegerVariable) solution.getVariable(0)).setValue(this.getPositionInArray("True", "True", "False"));
		((BinaryVariable) solution.getVariable(66)).set(0, false);
		((BinaryVariable) solution.getVariable(68)).set(0, false);
		((BinaryIntegerVariable) solution.getVariable(69)).setValue(this.getPositionInArray("True", "True", "False"));
		((BinaryIntegerVariable) solution.getVariable(70)).setValue(this.getPositionInArray("mutual_info", "mutual_info", "anova"));
		((BinaryIntegerVariable) solution.getVariable(71)).setValue(this.getPositionInArray("False", "True", "False"));
		((BinaryIntegerVariable) solution.getVariable(72)).setValue(5);

		SolutionDecoding solutionDecoding = genomeHandler.decodeGenome(solution);
		List<IComponentInstance> componentInstances = solutionDecoding.getComponentInstances();
		componentInstances.sort(new Comparator<IComponentInstance>() {
			@Override
			public int compare(final IComponentInstance c1, final IComponentInstance c2) {
				return c1.getComponent().getName().compareTo(c2.getComponent().getName());
			}
		});

		assertEquals(1, componentInstances.size());

		IComponentInstance rootComponentInstance = componentInstances.get(0);
		assertEquals("python_connection.feature_generation.tsfresh_feature_generator.TsFreshBasedFeatureExtractor", rootComponentInstance.getComponent().getName());

		IComponentInstance dictionaryComponentInstance = rootComponentInstance.getSatisfactionOfRequiredInterface("default_fc_parameters").get(0);
		assertEquals("python_connection.feature_generation.tsfresh_feature_generator.FCParametersDictionary", dictionaryComponentInstance.getComponent().getName());

		for (IParameter parameter : dictionaryComponentInstance.getComponent().getParameters()) {
			if (parameter.getName().equals("minimum")) {
				assertEquals("True", dictionaryComponentInstance.getParameterValue(parameter.getName()));
			} else {
				assertEquals("False", dictionaryComponentInstance.getParameterValue(parameter.getName()));
			}
		}
	}

}
