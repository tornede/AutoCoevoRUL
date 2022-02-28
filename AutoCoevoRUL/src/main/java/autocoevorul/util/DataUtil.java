package autocoevorul.util;

import java.lang.reflect.Field;
import java.util.Random;

import org.api4.java.ai.ml.core.dataset.schema.ILabeledInstanceSchema;
import org.api4.java.ai.ml.core.dataset.splitter.SplitFailedException;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.ai.ml.core.evaluation.execution.IDatasetSplitSet;

import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.ml.core.dataset.splitter.RandomHoldoutSplitter;
import ai.libs.jaicore.ml.core.evaluation.splitsetgenerator.CachingMonteCarloCrossValidationSplitSetGenerator;
import autocoevorul.experiment.ExperimentConfiguration;

public class DataUtil {

	public static IDatasetSplitSet<ILabeledDataset<?>> prepareDatasetSplits(final ExperimentConfiguration experimentConfiguration, final Random random)
			throws ExperimentEvaluationFailedException, IllegalAccessException, IllegalArgumentException, InterruptedException, NoSuchFieldException, SecurityException, SplitFailedException {
		ILabeledDataset<ILabeledInstance> trainingData = experimentConfiguration.getTrainingData();
		IDatasetSplitSet<ILabeledDataset<?>> datasetSplitSet = new CachingMonteCarloCrossValidationSplitSetGenerator<>(new RandomHoldoutSplitter<>(random, 0.7), experimentConfiguration.getNumberOfFolds(), random).nextSplitSet(trainingData);
		setDatasetRelationName(trainingData, datasetSplitSet, experimentConfiguration.getSeed());
		return datasetSplitSet;
	}

	private static void setDatasetRelationName(final ILabeledDataset<ILabeledInstance> data, final IDatasetSplitSet<ILabeledDataset<?>> datasetSplitSet, final long seed)
			throws NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException {
		ILabeledInstanceSchema instanceSchemaTemplate = data.getInstanceSchema().getCopy();

		for (int fold = 0; fold < datasetSplitSet.getNumberOfSplits(); fold++) {
			for (int i = 0; i < datasetSplitSet.getNumberOfFoldsPerSplit(); i++) {
				ILabeledDataset<?> dataset = datasetSplitSet.getFolds(fold).get(i);
				ILabeledInstanceSchema newInstanceSchema = instanceSchemaTemplate.getCopy();

				Field field = newInstanceSchema.getClass().getSuperclass().getDeclaredField("relationName");
				field.setAccessible(true);
				String newRelationName = dataset.getRelationName() + "_" + seed + "_" + fold;
				if (newRelationName.contains("train")) {
					newRelationName = newRelationName.replaceAll("_train", "");
				}
				if (newRelationName.contains("test")) {
					newRelationName = newRelationName.replaceAll("_test", "");
				}
				newRelationName += (i == 0 ? "_train" : "_test");
				field.set(newInstanceSchema, newRelationName);
				field.setAccessible(false);

				field = dataset.getClass().getDeclaredField("schema");
				field.setAccessible(true);
				field.set(dataset, newInstanceSchema);
				field.setAccessible(false);
			}
		}
	}

}
