package autocoevorul.featurerextraction.genomehandler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;

import org.moeaframework.core.Solution;
import org.moeaframework.core.Variable;
import org.moeaframework.core.variable.BinaryIntegerVariable;
import org.moeaframework.core.variable.BinaryVariable;
import org.moeaframework.core.variable.RealVariable;
import org.slf4j.Logger;

import ai.libs.jaicore.components.api.IComponent;
import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.components.api.IParameter;
import ai.libs.jaicore.components.api.IRequiredInterfaceDefinition;
import ai.libs.jaicore.components.exceptions.ComponentNotFoundException;
import ai.libs.jaicore.components.model.CategoricalParameterDomain;
import ai.libs.jaicore.components.model.ComponentInstance;
import ai.libs.jaicore.components.model.ComponentUtil;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.featurerextraction.FeatureExtractionMoeaProblem;
import autocoevorul.featurerextraction.GenomeComponentEntry;
import autocoevorul.featurerextraction.SolutionDecoding;
import autocoevorul.util.ComponentCollectionUtil;

public abstract class GenomeHandler {
	private static Logger LOGGER = org.slf4j.LoggerFactory.getLogger(GenomeHandler.class);

	protected List<IComponent> allComponents;
	protected List<IComponent> mainComponents;
	protected List<String> mainComponentNamesWithoutActivationBit;
	protected List<GenomeComponentEntry> mainComponentEntries;
	protected IComponent rootComponent;
	protected Map<Integer, GenomeComponentEntry> indexToComponentEntryMap;
	protected Map<String, List<GenomeComponentEntry>> providedInterfaceToListOfComponentEntryMap;
	protected List<Variable> variableTypes;
	protected int amountOfFeatures;

	protected GenomeHandler(final ExperimentConfiguration experimentConfiguration) throws IOException, ComponentNotFoundException, ExperimentEvaluationFailedException {
		this.allComponents = ComponentCollectionUtil.getAllComponents(experimentConfiguration.getFeatureSearchspace(), experimentConfiguration.getTemplateVariables());
		this.setupAdditionalComponents(experimentConfiguration);
		this.rootComponent = ComponentCollectionUtil.getComponentsSatisfyingInterface(experimentConfiguration.getFeatureSearchspace(), experimentConfiguration.getFeatureRequiredInterface(), experimentConfiguration.getTemplateVariables())
				.get(0);

		this.mainComponents = this.allComponents.stream().filter(c -> experimentConfiguration.getFeatureMainComponentNames().contains(c.getName())).sorted((c1, c2) -> c1.getName().compareTo(c2.getName())).collect(Collectors.toList());
		this.mainComponentNamesWithoutActivationBit = experimentConfiguration.getFeatureMainComponentNamesWithoutActivationBit();

		this.mainComponentEntries = new ArrayList<>();
		this.indexToComponentEntryMap = new HashMap<>();
		this.providedInterfaceToListOfComponentEntryMap = new HashMap<>();
		this.variableTypes = new ArrayList<>();

		for (IComponent component : this.allComponents.stream().sorted((c1, c2) -> c1.getName().compareTo(c2.getName())).collect(Collectors.toList())) {
			this.setupComponentToGenomeRepresentation(component);
		}
	}

	protected abstract void setupAdditionalComponents(final ExperimentConfiguration experimentConfiguration) throws ExperimentEvaluationFailedException;

	private void setupComponentToGenomeRepresentation(final IComponent component) throws ComponentNotFoundException {
		GenomeComponentEntry componentEntry;
		if (this.needsActivationBit(component)) {
			this.variableTypes.add(new BinaryVariable(1));
			componentEntry = new GenomeComponentEntry(this.amountOfFeatures, component.getName());
			this.indexToComponentEntryMap.put(this.amountOfFeatures, componentEntry);
			LOGGER.debug("({}) {}={}", this.amountOfFeatures, component.getName(), Arrays.asList(new String[] { "true", "false" }));
			this.amountOfFeatures++;
		} else {
			componentEntry = new GenomeComponentEntry(component.getName());
		}

		if (this.mainComponents.contains(component)) {
			this.mainComponentEntries.add(componentEntry);
		}

		for (String providedInterface : component.getProvidedInterfaces()) {
			if (!this.providedInterfaceToListOfComponentEntryMap.containsKey(providedInterface)) {
				this.providedInterfaceToListOfComponentEntryMap.put(providedInterface, new ArrayList<>());
			}
			this.providedInterfaceToListOfComponentEntryMap.get(providedInterface).add(componentEntry);
		}

		for (IParameter parameter : component.getParameters()) {
			if (componentEntry.addParameter(this.amountOfFeatures, parameter)) {
				this.indexToComponentEntryMap.put(this.amountOfFeatures, componentEntry);
				Variable variable = componentEntry.getParameterVariable(this.amountOfFeatures);
				this.variableTypes.add(variable);
				this.amountOfFeatures++;
			}
		}
	}

	private boolean needsActivationBit(final IComponent component) {
		if (this.mainComponents.contains(component)) {
			if (!this.mainComponentNamesWithoutActivationBit.contains(component.getName())) {
				return true;
			}
		} else if (component.getParameters().size() == 0 && component.getRequiredInterfaces().size() == 0) {
			return true;
		}
		return false;
	}

	public int getNumberOfVariables() {
		return this.amountOfFeatures;
	}

	public Solution newSolution() {
		Solution solution = new Solution(this.getNumberOfVariables(), 2);
		for (int i = 0; i < this.variableTypes.size(); i++) {
			Variable variable = this.variableTypes.get(i);
			if (variable != null) {
				solution.setVariable(i, variable.copy());
			}
		}
		return solution;
	}

	public SolutionDecoding decodeGenome(final Solution solution) throws ComponentNotFoundException {
		this.printGenome(solution);

		Map<String, List<IComponentInstance>> satisfactionOfRequiredInterfaces = new HashMap<>();
		for (IRequiredInterfaceDefinition requiredInterface : this.rootComponent.getRequiredInterfaces()) {
			satisfactionOfRequiredInterfaces.put(requiredInterface.getId(), new ArrayList<>());
			for (GenomeComponentEntry requiredInterfaceGenomeComponentEntry : this.providedInterfaceToListOfComponentEntryMap.get(requiredInterface.getName())) {
				IComponentInstance requiredInterfaceComponentInstance = this.decodeComponent(requiredInterfaceGenomeComponentEntry, solution);
				if (requiredInterfaceComponentInstance != null) {
					satisfactionOfRequiredInterfaces.get(requiredInterface.getId()).add(requiredInterfaceComponentInstance);
				}
			}
		}

		ComponentInstance componentInstance = new ComponentInstance(this.rootComponent, Collections.emptyMap(), satisfactionOfRequiredInterfaces);
		if (this.isCompletelyDefined(componentInstance)) {
			LOGGER.debug("ComponentInstances found: {}", componentInstance);
			return new SolutionDecoding(solution, componentInstance);
		}

		return null;
	}

	private IComponentInstance decodeComponent(final GenomeComponentEntry genomeComponentEntry, final Solution solution) throws ComponentNotFoundException {
		if (this.isComponentActivated(genomeComponentEntry, solution)) {
			IComponent component = ComponentUtil.getComponentByName(genomeComponentEntry.getName(), this.allComponents);

			Map<String, String> parameterValues = new HashMap<>();
			parameterValues.putAll(genomeComponentEntry.getSingleValueCategoricalParametersMap());
			for (Integer parameterIndex : genomeComponentEntry.getParameterIndices()) {
				IParameter parameter = genomeComponentEntry.getParameter(parameterIndex);
				String parameterValue = this.interpreteParameterVariable(solution, genomeComponentEntry, parameterIndex);
				parameterValues.put(parameter.getName(), parameterValue);
			}

			Map<String, List<IComponentInstance>> satisfactionOfRequiredInterfaces = new HashMap<>();
			for (IRequiredInterfaceDefinition requiredInterface : component.getRequiredInterfaces()) {
				satisfactionOfRequiredInterfaces.put(requiredInterface.getId(), new ArrayList<>());
				for (GenomeComponentEntry requiredInterfaceGenomeComponentEntry : this.providedInterfaceToListOfComponentEntryMap.get(requiredInterface.getName())) {
					IComponentInstance requiredInterfaceComponentInstance = this.decodeComponent(requiredInterfaceGenomeComponentEntry, solution);
					if (requiredInterfaceComponentInstance != null) {
						satisfactionOfRequiredInterfaces.get(requiredInterface.getId()).add(requiredInterfaceComponentInstance);
					}
				}
			}

			ComponentInstance componentInstance = new ComponentInstance(component, parameterValues, satisfactionOfRequiredInterfaces);
			if (this.isCompletelyDefined(componentInstance)) {
				LOGGER.debug("ComponentInstances found: {}", componentInstance);
				return componentInstance;
			}
		}
		return null;
	}

	private boolean isCompletelyDefined(final IComponentInstance componentInstance) {
		if (componentInstance.getParameterValues().size() != componentInstance.getComponent().getParameters().size()) {
			return false;
		}

		for (IRequiredInterfaceDefinition requiredInterface : componentInstance.getComponent().getRequiredInterfaces()) {
			if (requiredInterface.getMin() < componentInstance.getSatisfactionOfRequiredInterfaces().get(requiredInterface.getId()).size()
					&& componentInstance.getSatisfactionOfRequiredInterfaces().get(requiredInterface.getId()).size() < requiredInterface.getMax()) {
				return false;
			}
		}
		return true;
	}

	private void printGenome(final Solution solution) {
		List<GenomeComponentEntry> sortedGenomeComponentEntries = this.indexToComponentEntryMap.entrySet().stream().sorted(new Comparator<Entry<Integer, GenomeComponentEntry>>() {

			@Override
			public int compare(final Entry<Integer, GenomeComponentEntry> o1, final Entry<Integer, GenomeComponentEntry> o2) {
				return Integer.compare(o1.getKey(), o2.getKey());
			}
		}).map(e -> e.getValue()).distinct().collect(Collectors.toList());
		for (GenomeComponentEntry genomeComponentEntry : sortedGenomeComponentEntries) {
			if (genomeComponentEntry.hasActivationBit()) {
				LOGGER.debug("({}) {}={}", genomeComponentEntry.getActivationIndex(), genomeComponentEntry.getName(), ((BinaryVariable) solution.getVariable(genomeComponentEntry.getActivationIndex())).get(0));
			}
			for (Integer parameterIndex : genomeComponentEntry.getParameterIndices()) {
				IParameter parameter = genomeComponentEntry.getParameter(parameterIndex);
				String parameterValue = "";
				Variable variable = solution.getVariable(parameterIndex);
				if (variable instanceof BinaryIntegerVariable) {
					BinaryIntegerVariable binaryVariable = (BinaryIntegerVariable) variable;
					if (parameter.isCategorical()) {
						CategoricalParameterDomain categoricalParameter = (CategoricalParameterDomain) parameter.getDefaultDomain();
						parameterValue += categoricalParameter.getValues()[binaryVariable.getValue()];
					} else {
						parameterValue += binaryVariable.getValue();
					}

				} else if (variable instanceof RealVariable) {
					RealVariable numericVariable = (RealVariable) variable;
					parameterValue += numericVariable.getValue();
				}
				LOGGER.debug("({}) {}#{}={}", (parameterIndex), genomeComponentEntry.getName(), parameter.getName(), parameterValue);
			}
		}
	}

	protected boolean isComponentActivated(final GenomeComponentEntry componentEntry, final Solution solution) {
		if (componentEntry.hasActivationBit()) {
			return ((BinaryVariable) solution.getVariable(componentEntry.getActivationIndex())).get(0) == true;
		} else {
			for (Integer parameterIndex : componentEntry.getParameterIndices()) {
				if (this.interpreteParameterVariable(solution, componentEntry, parameterIndex).toLowerCase().equals("true")) {
					return true;
				}
			}
		}
		return false;
	}

	protected String interpreteParameterVariable(final Solution solution, final GenomeComponentEntry componentEntry, final Integer parameterIndex) {
		IParameter parameter = componentEntry.getParameter(parameterIndex);
		String parameterValue = "";
		Variable variable = solution.getVariable(parameterIndex);
		if (variable instanceof BinaryIntegerVariable) {
			BinaryIntegerVariable binaryVariable = (BinaryIntegerVariable) variable;
			if (parameter.isCategorical()) {
				CategoricalParameterDomain categoricalParameter = (CategoricalParameterDomain) parameter.getDefaultDomain();
				parameterValue += categoricalParameter.getValues()[binaryVariable.getValue()];
			} else {
				parameterValue += binaryVariable.getValue();
			}

		} else if (variable instanceof RealVariable) {
			RealVariable numericVariable = (RealVariable) variable;
			parameterValue += numericVariable.getValue();
		}

		return parameterValue;
	}

	public int getIndexOfComponent(final String componentName) {
		for (Entry<Integer, GenomeComponentEntry> entry : this.indexToComponentEntryMap.entrySet()) {
			if (entry.getValue().getName().endsWith(componentName)) {
				return entry.getKey();
			}
		}
		return -1;
	}

	public Set<Integer> getIndexOfParameters(final String componentName) {
		for (Entry<Integer, GenomeComponentEntry> entry : this.indexToComponentEntryMap.entrySet()) {
			if (entry.getValue().getName().endsWith(componentName)) {
				return entry.getValue().getParameterIndices();
			}
		}
		return null;
	}

	public int getIndexOfParameter(final String componentName, final String parameterName) {
		for (Entry<Integer, GenomeComponentEntry> entry : this.indexToComponentEntryMap.entrySet()) {
			if (entry.getValue().getName().endsWith(componentName)) {
				for (Integer index : entry.getValue().getParameterIndices()) {
					IParameter parameter = entry.getValue().getParameter(index);
					if (parameter.getName().equals(parameterName)) {
						return index;
					}
				}
			}
		}
		return -1;
	}

	public abstract List<Solution> getInitialSolutions(final FeatureExtractionMoeaProblem problem, final int populationSize) throws ComponentNotFoundException;

	public abstract Solution getEmptySolution(final FeatureExtractionMoeaProblem problem);

	public abstract Solution activateTsfresh(Solution solution);

	protected static int getPositionInArray(final Object value, final Object[] array) {
		for (int i = 0; i < array.length; i++) {
			if (array[i].equals(value)) {
				return i;
			}
		}
		return -1;
	}

}
