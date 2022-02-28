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
		for (IAttribute attribute : experimentConfiguration.getTrainingData().getListOfAttributes()) {
			String genome_name = this.getAttributeNameForGenomeRepresentation(attribute);
			this.allComponents.add(new Component(genome_name, Arrays.asList("AttributeName"), new ArrayList<IRequiredInterfaceDefinition>(), new PartialOrderedSet<>(), new ArrayList<>()));
		}
	}

	private String getAttributeNameForGenomeRepresentation(final IAttribute attribute) {
		return "attribute_" + attribute.getName();
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
			if (componentEntry.getName().contains("TsfreshWrapper")) {
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
