package autocoevorul.featurerextraction;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.StringJoiner;
import java.util.stream.Collectors;

import org.moeaframework.core.Solution;

import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.mlplan.sklearn.ScikitLearnClassifierFactory;

public class SolutionDecoding {

	private Solution solution;
	private List<IComponentInstance> componentInstances;
	private String constructionInstruction;
	private String imports;

	public SolutionDecoding(final Solution solution, final List<IComponentInstance> componentInstances) {
		super();
		this.solution = solution;
		this.componentInstances = componentInstances;

		ScikitLearnClassifierFactory factory = new ScikitLearnClassifierFactory();
		Set<String> importSet = new HashSet<>();
		StringJoiner constructionStringJoiner = new StringJoiner(",");
		for (IComponentInstance componentInstance : componentInstances) {
			constructionStringJoiner.add(factory.extractSKLearnConstructInstruction(componentInstance, importSet));
		}
		this.constructionInstruction = constructionStringJoiner.toString();
		if (componentInstances.size() > 1) {
			this.constructionInstruction = "make_union(" + this.constructionInstruction + ")";
		}

		StringBuilder importsStringBuilder = new StringBuilder();
		for (String importString : importSet.stream().sorted().collect(Collectors.toList())) {
			importsStringBuilder.append(importString);
		}
		this.imports = importsStringBuilder.toString();
	}

	public void setPerformance(final double performance, final double numberOfUsage) {
		this.solution.setObjective(0, performance);
		this.solution.setObjective(1, numberOfUsage);
	}

	public Solution getSolution() {
		return this.solution;
	}

	public List<IComponentInstance> getComponentInstances() {
		return this.componentInstances;
	}

	public String getConstructionInstruction() {
		return this.constructionInstruction;
	}

	public String getImports() {
		return this.imports;
	}

}
