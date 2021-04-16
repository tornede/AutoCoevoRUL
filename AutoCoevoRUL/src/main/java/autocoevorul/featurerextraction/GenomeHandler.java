package autocoevorul.featurerextraction;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
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
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.util.ComponentCollectionUtil;

public class GenomeHandler {

	private static Logger LOGGER = org.slf4j.LoggerFactory.getLogger(GenomeHandler.class);

	private List<IComponent> allComponents;
	private List<IComponent> mainComponents;
	private List<String> mainComponentNamesWithoutActivationBit;
	private List<GenomeComponentEntry> mainComponentEntries;
	private List<IComponent> rootComponents;
	private Map<Integer, GenomeComponentEntry> indexToComponentEntryMap;
	private Map<String, List<GenomeComponentEntry>> requiredInterfaceToListOfComponentEntryMap;
	private Map<String, List<GenomeComponentEntry>> providedInterfaceToListOfComponentEntryMap;
	private List<Variable> variableTypes;
	private int amountOfFeatures;

	// final String searchSpaceFile, final String requiredRootInterface, final Map<String, String> templateVariables, final List<String> mainComponentNames,
	// final List<String> mainComponentNamesWithoutActivationBit
	public GenomeHandler(final ExperimentConfiguration experimentConfiguration) throws IOException, ComponentNotFoundException {
		this.allComponents = ComponentCollectionUtil.getAllComponents(experimentConfiguration.getFeatureSearchspace(), experimentConfiguration.getTemplateVariables());
		this.rootComponents = ComponentCollectionUtil.getComponentsSatisfyingInterface(experimentConfiguration.getFeatureSearchspace(), experimentConfiguration.getFeatureRequiredInterface(),
				experimentConfiguration.getTemplateVariables());

		this.mainComponents = this.allComponents.stream().filter(c -> experimentConfiguration.getFeatureMainComponentNames().contains(c.getName()))
				.sorted((c1, c2) -> c1.getName().compareTo(c2.getName())).collect(Collectors.toList());
		this.mainComponentNamesWithoutActivationBit = experimentConfiguration.getFeatureMainComponentNamesWithoutActivationBit();

		this.mainComponentEntries = new ArrayList<>();
		this.indexToComponentEntryMap = new HashMap<>();
		this.requiredInterfaceToListOfComponentEntryMap = new HashMap<>();
		this.providedInterfaceToListOfComponentEntryMap = new HashMap<>();
		this.variableTypes = new ArrayList<>();
		for (IComponent component : this.allComponents) {
			this.setupComponentToGenomeRepresentation(component);
		}
	}

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

		this.addComponentEntryForRequiredInterfaces(component.getRequiredInterfaces(), componentEntry);
		this.addComponentEntryForProvidedInterfaces(component.getProvidedInterfaces(), componentEntry);

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

	private void addComponentEntryForRequiredInterfaces(final Collection<IRequiredInterfaceDefinition> requiredInterfaces, final GenomeComponentEntry componentEntry) {
		for (IRequiredInterfaceDefinition requiredInterface : requiredInterfaces) {
			if (!this.requiredInterfaceToListOfComponentEntryMap.containsKey(requiredInterface.getName())) {
				this.requiredInterfaceToListOfComponentEntryMap.put(requiredInterface.getName(), new ArrayList<>());
			}
			this.requiredInterfaceToListOfComponentEntryMap.get(requiredInterface.getName()).add(componentEntry);
		}
	}

	private void addComponentEntryForProvidedInterfaces(final Collection<String> providedInterfaces, final GenomeComponentEntry componentEntry) {
		for (String providedInterface : providedInterfaces) {
			if (!this.providedInterfaceToListOfComponentEntryMap.containsKey(providedInterface)) {
				this.providedInterfaceToListOfComponentEntryMap.put(providedInterface, new ArrayList<>());
			}
			this.providedInterfaceToListOfComponentEntryMap.get(providedInterface).add(componentEntry);
		}
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
		List<IComponentInstance> componentInstances = new ArrayList<>();
		for (GenomeComponentEntry componentEntry : this.mainComponentEntries) {
			IComponentInstance mainComponentInstance = this.decodeMainComponent(componentEntry, solution);
			while (mainComponentInstance != null && !this.rootComponents.contains(mainComponentInstance.getComponent())) {
				for (String providedInterface : mainComponentInstance.getComponent().getProvidedInterfaces()) {
					if (this.requiredInterfaceToListOfComponentEntryMap.containsKey(providedInterface)) {
						for (GenomeComponentEntry x : this.requiredInterfaceToListOfComponentEntryMap.get(providedInterface)) {
							mainComponentInstance = this.decodeParentComponent(solution, x, providedInterface, mainComponentInstance);
						}
					}
				}
			}
			if (mainComponentInstance != null) {
				componentInstances.add(mainComponentInstance);
				LOGGER.debug("ComponentInstance found: {}", mainComponentInstance);
			}
		}

		if (componentInstances.size() > 0) {
			LOGGER.debug("ComponentInstances found: {}", componentInstances.size());
			return new SolutionDecoding(solution, componentInstances);
		}
		return null;

	}

	private IComponentInstance decodeMainComponent(final GenomeComponentEntry componentEntry, final Solution solution) throws ComponentNotFoundException {
		if (this.isComponentActivated(componentEntry, solution)) {
			IComponent component = ComponentUtil.getComponentByName(componentEntry.getName(), this.allComponents);

			Map<String, String> parameterValues = new HashMap<>();
			parameterValues.putAll(componentEntry.getSingleValueCategoricalParametersMap());
			for (Integer parameterIndex : componentEntry.getParameterIndices()) {
				IParameter parameter = componentEntry.getParameter(parameterIndex);
				String parameterValue = this.interpreteParameterVariable(solution, componentEntry, parameterIndex);
				parameterValues.put(parameter.getName(), parameterValue);
			}

			ComponentInstance componentInstance = new ComponentInstance(component, parameterValues, new HashMap<>());
			if (this.isCompletelyDefined(componentInstance)) {
				return componentInstance;
			}
		}
		return null;
	}

	private IComponentInstance decodeParentComponent(final Solution solution, final GenomeComponentEntry componentEntry, final String fixedRequiredInterface,
			final IComponentInstance childComponentInstance) throws ComponentNotFoundException {
		IComponent component = ComponentUtil.getComponentByName(componentEntry.getName(), this.allComponents);

		Map<String, String> parameterValues = new HashMap<>();
		parameterValues.putAll(componentEntry.getSingleValueCategoricalParametersMap());
		for (Integer parameterIndex : componentEntry.getParameterIndices()) {
			IParameter parameter = componentEntry.getParameter(parameterIndex);
			String parameterValue = this.interpreteParameterVariable(solution, componentEntry, parameterIndex);
			parameterValues.put(parameter.getName(), parameterValue);
		}

		Map<String, List<IComponentInstance>> satisfactionOfRequiredInterfaces = new HashMap<>();
		for (IRequiredInterfaceDefinition requiredInterface : component.getRequiredInterfaces()) {
			if (!satisfactionOfRequiredInterfaces.containsKey(requiredInterface.getId())) {
				satisfactionOfRequiredInterfaces.put(requiredInterface.getId(), new ArrayList<>());
			}
			if (requiredInterface.getName().equals(fixedRequiredInterface)) {
				satisfactionOfRequiredInterfaces.get(requiredInterface.getId()).add(childComponentInstance);
			} else {
				for (GenomeComponentEntry possibleComponentEntry : this.providedInterfaceToListOfComponentEntryMap.get(requiredInterface.getName())) {
					IComponentInstance possibleComponentInstance = this.decodeComponent(solution, possibleComponentEntry);
					if (possibleComponentInstance != null) {
						satisfactionOfRequiredInterfaces.get(requiredInterface.getId()).add(possibleComponentInstance);
					}
				}
			}
		}

		ComponentInstance componentInstance = new ComponentInstance(component, parameterValues, satisfactionOfRequiredInterfaces);
		if (this.isCompletelyDefined(componentInstance)) {
			return componentInstance;
		}
		return null;
	}

	private IComponentInstance decodeComponent(final Solution solution, final GenomeComponentEntry componentEntry) throws ComponentNotFoundException {
		IComponent component = ComponentUtil.getComponentByName(componentEntry.getName(), this.allComponents);
		if (this.isComponentActivated(componentEntry, solution) && !this.rootComponents.contains(component)) {
			Map<String, String> parameterValues = new HashMap<>();
			parameterValues.putAll(componentEntry.getSingleValueCategoricalParametersMap());
			for (Integer parameterIndex : componentEntry.getParameterIndices()) {
				IParameter parameter = componentEntry.getParameter(parameterIndex);
				String parameterValue = this.interpreteParameterVariable(solution, componentEntry, parameterIndex);
				parameterValues.put(parameter.getName(), parameterValue);
			}
			Map<String, List<IComponentInstance>> satisfactionOfRequiredInterfaces = new HashMap<>();
			for (IRequiredInterfaceDefinition requiredInterface : component.getRequiredInterfaces()) {
				if (!satisfactionOfRequiredInterfaces.containsKey(requiredInterface.getId())) {
					satisfactionOfRequiredInterfaces.put(requiredInterface.getId(), new ArrayList<>());
				}
				for (GenomeComponentEntry possibleComponentEntry : this.providedInterfaceToListOfComponentEntryMap.get(requiredInterface.getName())) {
					IComponentInstance possibleComponentInstance = this.decodeComponent(solution, possibleComponentEntry);
					if (possibleComponentInstance != null) {
						satisfactionOfRequiredInterfaces.get(requiredInterface.getId()).add(possibleComponentInstance);
					}
				}
			}

			ComponentInstance componentInstance = new ComponentInstance(component, parameterValues, satisfactionOfRequiredInterfaces);
			if (this.isCompletelyDefined(componentInstance)) {
				return componentInstance;
			}
		}
		return null;
	}

	private boolean isCompletelyDefined(final IComponentInstance componentInstance) {
		boolean isCompletelyDefined = true;
		if (componentInstance.getParameterValues().size() != componentInstance.getComponent().getParameters().size()) {
			isCompletelyDefined = false;
		}

		for (IRequiredInterfaceDefinition requiredInterface : componentInstance.getComponent().getRequiredInterfaces()) {
			if (componentInstance.getSatisfactionOfRequiredInterfaces().get(requiredInterface.getId()).size() < 1) {
				isCompletelyDefined = false;
			}
		}
		return isCompletelyDefined;
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
				LOGGER.trace("({}) {}={}", genomeComponentEntry.getActivationIndex(), genomeComponentEntry.getName(),
						((BinaryVariable) solution.getVariable(genomeComponentEntry.getActivationIndex())).get(0));
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
				LOGGER.trace("({}) {}#{}={}", (parameterIndex), genomeComponentEntry.getName(), parameter.getName(), parameterValue);
			}
		}
	}

	private boolean isComponentActivated(final GenomeComponentEntry componentEntry, final Solution solution) {
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

	private String interpreteParameterVariable(final Solution solution, final GenomeComponentEntry componentEntry, final Integer parameterIndex) {
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

}
