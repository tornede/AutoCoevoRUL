package autocoevorul.baseline.randomsearch;

import java.io.IOException;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.stream.Collectors;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.execution.IDatasetSplitSet;
import org.api4.java.common.attributedobjects.ObjectEvaluationFailedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.eventbus.EventBus;

import ai.libs.jaicore.components.api.IComponent;
import ai.libs.jaicore.components.api.IComponentInstance;
import ai.libs.jaicore.components.model.ComponentRepository;
import ai.libs.jaicore.components.model.ComponentUtil;
import autocoevorul.SearchResult;
import autocoevorul.experiment.ExperimentConfiguration;
import autocoevorul.util.ComponentCollectionUtil;

public class RandomSearch {

	protected static final Logger LOGGER = LoggerFactory.getLogger(RandomSearch.class);

	private EventBus eventBus;
	private final ExperimentConfiguration experimentConfiguration;
	private final Random random;
	private final IDatasetSplitSet<ILabeledDataset<?>> datasetSplitSet;
	private final List<List<? extends Double>> groundTruthsForSplits;

	private List<IComponent> allComponents;
	private SearchResult bestPipeline;

	public RandomSearch(final EventBus eventBus, final ExperimentConfiguration experimentConfiguration, final IDatasetSplitSet<ILabeledDataset<?>> datasetSplitSet) throws SQLException, IOException {
		this.eventBus = eventBus;
		this.experimentConfiguration = experimentConfiguration;
		this.random = new Random(experimentConfiguration.getSeed());

		this.datasetSplitSet = datasetSplitSet;
		this.groundTruthsForSplits = new ArrayList<>(experimentConfiguration.getNumberOfFolds());
		for (int split = 0; split < datasetSplitSet.getNumberOfSplits(); split++) {
			this.groundTruthsForSplits.add(this.datasetSplitSet.getFolds(split).get(1).stream().map(instance -> (Double) instance.getLabel()).collect(Collectors.toList()));
		}

		this.allComponents = ComponentCollectionUtil.getAllComponents(experimentConfiguration.getRulSearchSpace(), experimentConfiguration.getTemplateVariables());
		this.bestPipeline = new SearchResult(eventBus);
	}

	public PipelineEvaluationReport run() throws ObjectEvaluationFailedException, InterruptedException, IOException {
		LOGGER.info("Start RandomSearch");
		long start = System.currentTimeMillis();

		ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(this.experimentConfiguration.getNumCPUs());
		while ((System.currentTimeMillis() - start) < this.experimentConfiguration.getTotalTimeout().milliseconds()) {
			if (executor.getQueue().size() < this.experimentConfiguration.getNumCPUs()) {
				IComponentInstance randomComponentInstance = ComponentUtil.getRandomInstantiationOfComponent(this.experimentConfiguration.getRulRootComponentName(), new ComponentRepository(this.allComponents), this.random);

				RulCompletePipelineEvaluator evaluator = new RulCompletePipelineEvaluator(this.eventBus, this.experimentConfiguration, this.datasetSplitSet, this.experimentConfiguration.getRulTimeout(), randomComponentInstance,
						this.bestPipeline);
				executor.execute(evaluator);
			}
		}

		LOGGER.info("End RandomSearch. Maximum threads inside pool: " + executor.getMaximumPoolSize());
		executor.shutdownNow();

		return this.bestPipeline.getPipelineEvaluationReport();
	}

}
