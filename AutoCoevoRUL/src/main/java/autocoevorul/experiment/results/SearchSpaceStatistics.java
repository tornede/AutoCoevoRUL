package autocoevorul.experiment.results;

import java.io.IOException;

import ai.libs.jaicore.components.api.IComponent;
import ai.libs.jaicore.components.api.IParameter;
import ai.libs.jaicore.components.model.CategoricalParameterDomain;
import ai.libs.jaicore.components.model.NumericParameterDomain;
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.util.ComponentCollectionUtil;

public class SearchSpaceStatistics {

	public static void main(final String[] args) throws IOException {
		ExperimentConfiguration config = new ExperimentConfiguration();
		int catParameters = 0;
		int numParameters = 0;
		for (IComponent component : ComponentCollectionUtil.getAllComponents(config.getRegressionSearchpace(), config.getTemplateVariables())) {
			System.out.println("\n" + component.getName());
			for (IParameter parameter : component.getParameters()) {
				System.out.print(parameter.getName());
				if (parameter.isCategorical()) {
					CategoricalParameterDomain categoricalParameter = (CategoricalParameterDomain) parameter.getDefaultDomain();
					if (categoricalParameter.getValues().length > 1) {
						catParameters++;
						System.out.println(" yes");
					} else {
						System.out.println("--");
					}
				} else if (parameter.isNumeric()) {
					NumericParameterDomain numericParameter = (NumericParameterDomain) parameter.getDefaultDomain();
					if (numericParameter.getMin() < numericParameter.getMax()) {
						numParameters++;
						System.out.println(" yes");
					} else {
						System.out.println("--");
					}
				}
			}
		}

		System.out.println("\nCat: " + catParameters + "\nNum: " + numParameters);
	}
}
