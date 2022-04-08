package autocoevorul.util;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.sets.Pair;
import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.mlplan.sklearn.AScikitLearnLearnerFactory;

public class ScikitLearnUtil {

	public static Pair<String, String> createConstructionInstructionAndImportsFromComponentInstance(final AScikitLearnLearnerFactory factory, final IComponentInstance componentInstance) {
		Set<String> importSet = new HashSet<>();
		String constructionString = factory.extractSKLearnConstructInstruction(componentInstance, importSet);

		StringBuilder imports = new StringBuilder();
		for (String importString : importSet.stream().sorted().collect(Collectors.toList())) {
			imports.append(importString);
		}
		return new Pair<>(constructionString, imports.toString());
	}

}
