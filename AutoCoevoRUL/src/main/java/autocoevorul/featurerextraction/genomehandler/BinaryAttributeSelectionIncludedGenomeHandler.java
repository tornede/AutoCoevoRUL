package autocoevorul.featurerextraction.genomehandler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.api4.java.ai.ml.core.dataset.schema.attribute.IAttribute;
import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.BinaryIntegerVariable;
import org.moeaframework.core.variable.BinaryVariable;
import org.slf4j.Logger;

import ai.libs.jaicore.basic.sets.PartialOrderedSet;
import ai.libs.jaicore.components.api.IComponent;
import ai.libs.jaicore.components.api.IParameter;
import ai.libs.jaicore.components.api.IRequiredInterfaceDefinition;
import ai.libs.jaicore.components.exceptions.ComponentNotFoundException;
import ai.libs.jaicore.components.model.CategoricalParameterDomain;
import ai.libs.jaicore.components.model.Component;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.featurerextraction.FeatureExtractionMoeaProblem;
import autocoevorul.featurerextraction.GenomeComponentEntry;

public class BinaryAttributeSelectionIncludedGenomeHandler extends GenomeHandler {

	private static Logger LOGGER = org.slf4j.LoggerFactory.getLogger(BinaryAttributeSelectionIncludedGenomeHandler.class);

	public BinaryAttributeSelectionIncludedGenomeHandler(final ExperimentConfiguration experimentConfiguration) throws IOException, ComponentNotFoundException, ExperimentEvaluationFailedException {
		super(experimentConfiguration);
	}

	@Override
	protected void setupAdditionalComponents(final ExperimentConfiguration experimentConfiguration) throws ExperimentEvaluationFailedException {
		for (int a = 0; a < experimentConfiguration.getTrainingData().getListOfAttributes().size(); a++) {
			IAttribute attribute = experimentConfiguration.getTrainingData().getListOfAttributes().get(a);
			String genome_name = "ml4pdm.transformation.AttributeFilter." + a + "." + attribute.getName();
			this.allComponents.add(new Component(genome_name, Arrays.asList("AttributeId"), new ArrayList<IRequiredInterfaceDefinition>(), new PartialOrderedSet<>(), new ArrayList<>()));
		}
	}

	@Override
	public List<Solution> getInitialSolutions(final FeatureExtractionMoeaProblem problem, final int populationSize) throws ComponentNotFoundException {
		List<Solution> initialSolutions = new ArrayList<>();

		initialSolutions.add(this.activateTsfresh(this.getEmptySolution(problem)));

		return initialSolutions;
	}

	@Override
	public Solution getEmptySolution(final FeatureExtractionMoeaProblem problem) {
		Solution solution = problem.newSolution();
		for (IComponent component : this.mainComponents) {
			if (!this.mainComponentNamesWithoutActivationBit.contains(component.getName())) {
				((BinaryVariable) solution.getVariable(this.getIndexOfComponent(component.getName()))).set(0, false);
			}
			for (IParameter parameter : component.getParameters()) {
				int parameterIndex = this.getIndexOfParameter(component.getName(), parameter.getName());
				if (parameter.getDefaultDomain() instanceof CategoricalParameterDomain) {
					((BinaryVariable) solution.getVariable(parameterIndex)).set(0, false);
				} else {
					double value = (double) parameter.getDefaultValue();
					((BinaryIntegerVariable) solution.getVariable(parameterIndex)).setValue((int) value);
				}
			}
		}
		return solution;
	}

	public Solution activateAttributeFilter(final Solution solution) {
		List<Integer> defaultAttributesToFilter = Arrays.asList(1, 2);
		for (IComponent component : this.allComponents.stream().filter(c -> c.getName().contains("AttributeFilter.")).sorted((c1, c2) -> c1.getName().compareTo(c2.getName())).collect(Collectors.toList())) {
			int attributeID = Integer.parseInt(component.getName().split("\\.")[3]);
			if (defaultAttributesToFilter.contains(attributeID)) {
				((BinaryVariable) solution.getVariable(this.getIndexOfComponent(component.getName()))).set(0, true);
			}
		}
		return solution;
	}

	@Override
	public Solution activateTsfresh(final Solution solution) {
		List<String> defaultTsfreshFeatures = Arrays.asList(new String[] { "MAXIMUM", "MINIMUM" });
		for (IComponent component : this.allComponents.stream().filter(c -> c.getName().contains("TsfreshFeature")).sorted((c1, c2) -> c1.getName().compareTo(c2.getName())).collect(Collectors.toList())) {
			String tsfreshFeature = component.getName().substring(component.getName().lastIndexOf('.') + 1);
			if (defaultTsfreshFeatures.contains(tsfreshFeature)) {
				((BinaryVariable) solution.getVariable(this.getIndexOfComponent(component.getName()))).set(0, true);
			}
		}
		return solution;
	}

	@Override
	protected boolean isComponentActivated(final GenomeComponentEntry componentEntry, final Solution solution) {
		if (componentEntry.hasActivationBit()) {
			return ((BinaryVariable) solution.getVariable(componentEntry.getActivationIndex())).get(0) == true;
		} else {
			if (componentEntry.getName().contains("AttributeFilter")) {
				for (GenomeComponentEntry attributeFilterGenomeComponentEntry : this.providedInterfaceToListOfComponentEntryMap.get("AttributeId")) {
					if (this.isComponentActivated(attributeFilterGenomeComponentEntry, solution)) {
						return true;
					}
				}
			} else if (componentEntry.getName().contains("TsfreshWrapper")) {
				for (GenomeComponentEntry tsfreshFeatureGenomeComponentEntry : this.providedInterfaceToListOfComponentEntryMap.get("TsfreshFeature")) {
					if (this.isComponentActivated(tsfreshFeatureGenomeComponentEntry, solution)) {
						return true;
					}
				}
			}

			for (Integer parameterIndex : componentEntry.getParameterIndices()) {
				if (this.interpreteParameterVariable(solution, componentEntry, parameterIndex).toLowerCase().equals("true")) {
					return true;
				}
			}
		}
		return false;
	}

}
