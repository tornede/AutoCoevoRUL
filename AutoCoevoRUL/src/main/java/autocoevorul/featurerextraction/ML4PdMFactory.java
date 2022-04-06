package autocoevorul.featurerextraction;

import java.io.IOException;
import java.util.Set;
import java.util.StringJoiner;
import java.util.stream.Collectors;

import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.mlplan.sklearn.AScikitLearnLearnerFactory;

public class ML4PdMFactory extends AScikitLearnLearnerFactory {

	public ML4PdMFactory() {
		super();
	}

	@Override
	public String getPipelineBuildString(final IComponentInstance groundComponent, final Set<String> importSet) {
		StringJoiner sb = new StringJoiner(",");
		for (IComponentInstance timeseriesFeatureGenerator : groundComponent.getSatisfactionOfRequiredInterface("attribute_filter")) {
			sb.add(this.getPipelineBuildStringForAttributeFilter(timeseriesFeatureGenerator, importSet));
		}
		for (IComponentInstance componentInstances : groundComponent.getSatisfactionOfRequiredInterface("timeseries_feature_generator").stream().sorted((o1, o2) -> o1.getComponent().getName().compareTo(o2.getComponent().getName()))
				.collect(Collectors.toList())) {
			sb.add(this.getPipelineBuildStringForTimeseriesFeatureExtraction(componentInstances, importSet));
		}
		return sb.toString();
	}

	private String getPipelineBuildStringForAttributeFilter(final IComponentInstance componentInstances, final Set<String> importSet) {
		StringBuilder sb = new StringBuilder();
		if (componentInstances.getComponent().getName().startsWith("ml4pdm.transformation.AttributeFilter")) {
			this.extractImport(componentInstances, importSet);
			StringJoiner attribute_ids_string = new StringJoiner(",");
			for (IComponentInstance satisfyingComponentInstance : componentInstances.getSatisfactionOfRequiredInterface("attribute_ids").stream().sorted((o1, o2) -> o1.getComponent().getName().compareTo(o2.getComponent().getName()))
					.collect(Collectors.toList())) {
				String[] attribute_name_splitted = satisfyingComponentInstance.getComponent().getName().split("\\.");
				String attribute_id = attribute_name_splitted[attribute_name_splitted.length - 2];
				attribute_ids_string.add(attribute_id);
			}
			sb.append("AttributeFilter(attribute_ids=[");
			sb.append(attribute_ids_string.toString());
			sb.append("])");
		} else {
			sb.append(this.extractSKLearnConstructInstruction(componentInstances, importSet));
		}

		return sb.toString();
	}

	private String getPipelineBuildStringForTimeseriesFeatureExtraction(final IComponentInstance componentInstances, final Set<String> importSet) {
		StringBuilder sb = new StringBuilder();
		if (componentInstances.getComponent().getName().startsWith("ml4pdm.transformation.AttributeFilter")) {
			this.extractImport(componentInstances, importSet);
			StringJoiner sj = new StringJoiner(",");
			for (IComponentInstance satisfyingComponentInstance : componentInstances.getSatisfactionOfRequiredInterface("attribute_ids").stream().sorted((o1, o2) -> o1.getComponent().getName().compareTo(o2.getComponent().getName()))
					.collect(Collectors.toList())) {
				this.extractImport(satisfyingComponentInstance, importSet, 2);
				String[] packagePathSplit = satisfyingComponentInstance.getComponent().getName().split("\\.");
				String tsfreshFeature = packagePathSplit[packagePathSplit.length - 2] + "." + packagePathSplit[packagePathSplit.length - 1];
				sj.add(tsfreshFeature);
			}
			sb.append("TsfreshWrapper(tsfresh_features=[");
			sb.append(sj.toString());
			sb.append("])");
		} else if (componentInstances.getComponent().getName().endsWith("TsfreshWrapper")) {
			this.extractImport(componentInstances, importSet);
			StringJoiner sj = new StringJoiner(",");
			for (IComponentInstance satisfyingComponentInstance : componentInstances.getSatisfactionOfRequiredInterface("tsfresh_features").stream().sorted((o1, o2) -> o1.getComponent().getName().compareTo(o2.getComponent().getName()))
					.collect(Collectors.toList())) {
				this.extractImport(satisfyingComponentInstance, importSet, 2);
				String[] packagePathSplit = satisfyingComponentInstance.getComponent().getName().split("\\.");
				String tsfreshFeature = packagePathSplit[packagePathSplit.length - 2] + "." + packagePathSplit[packagePathSplit.length - 1];
				sj.add(tsfreshFeature);
			}
			sb.append("TsfreshWrapper(tsfresh_features=[");
			sb.append(sj.toString());
			sb.append("])");
		} else {
			sb.append(this.extractSKLearnConstructInstruction(componentInstances, importSet));
		}

		return sb.toString();
	}

	private void extractImport(final IComponentInstance componentInstances, final Set<String> importSet) {
		this.extractImport(componentInstances, importSet, 1);
	}

	private void extractImport(final IComponentInstance componentInstances, final Set<String> importSet, final int classNameindex) {
		String[] packagePathSplit = componentInstances.getComponent().getName().split("\\.");
		StringBuilder fromSB = new StringBuilder();
		fromSB.append(packagePathSplit[0]);
		for (int i = 1; i < packagePathSplit.length - classNameindex; i++) {
			fromSB.append("." + packagePathSplit[i]);
		}
		String className = packagePathSplit[packagePathSplit.length - classNameindex];

		if (packagePathSplit.length > 1) {
			importSet.add("from " + fromSB.toString() + " import " + className + "\n");
		}
	}

	@Override
	public ML4PdMTimeSeriesFeatureEngineeringWrapper getScikitLearnWrapper(final String constructionString, final String imports) throws IOException, InterruptedException {
		return new ML4PdMTimeSeriesFeatureEngineeringWrapper(constructionString, imports);
	}
}
