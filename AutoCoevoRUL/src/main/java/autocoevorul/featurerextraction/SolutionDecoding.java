package autocoevorul.featurerextraction;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

import org.moeaframework.core.Solution;

import ai.libs.jaicore.components.api.IComponentInstance;

public class SolutionDecoding {

	private Solution solution;
	private IComponentInstance componentInstance;
	private String constructionInstruction;
	private String imports;

	public SolutionDecoding(final Solution solution, final IComponentInstance componentInstances) {
		super();
		this.solution = solution;
		this.componentInstance = componentInstances;

		TimeseriesFeatureEngineeringScikitLearnFactory factory = new TimeseriesFeatureEngineeringScikitLearnFactory(); // TODO
		
		Set<String> importSet = new HashSet<>();
		this.constructionInstruction = factory.extractSKLearnConstructInstruction(componentInstance, importSet);

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

	public IComponentInstance getComponentInstance() {
		return this.componentInstance;
	}

	public String getConstructionInstruction() {
		return this.constructionInstruction;
	}

	public String getImports() {
		return this.imports;
	}

}
