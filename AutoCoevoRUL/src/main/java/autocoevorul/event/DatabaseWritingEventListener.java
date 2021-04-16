package autocoevorul.event;

import java.sql.Date;
import java.sql.SQLException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang.exception.ExceptionUtils;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.ai.ml.core.evaluation.execution.ILearnerRunReport;
import org.api4.java.datastructure.kvstore.IKVStore;

import com.google.common.eventbus.Subscribe;

import ai.libs.jaicore.basic.kvstore.KVStore;
import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.DatabaseAdapterFactory;
import ai.libs.jaicore.ml.core.evaluation.evaluator.events.TrainTestSplitEvaluationFailedEvent;
import ai.libs.jaicore.ml.regression.singlelabel.SingleTargetRegressionPrediction;
import ai.libs.jaicore.ml.regression.singlelabel.SingleTargetRegressionPredictionBatch;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnWrapper;
import ai.libs.mlplan.core.events.ClassifierFoundEvent;
import autocoevorul.experiment.ExperimentConfiguration;

public class DatabaseWritingEventListener {

	private static final String FIELD_TYPE_INT = "INT";
	private static final String FIELD_PRIMARY_KEY = "primary";
	private static final String FIELD_EXPERIMENT_ID = "experiment_id";
	private static final String FIELD_EXECUTOR = "executor";
	private static final String FIELD_THREAD = "thread";
	private static final String FIELD_DATASET = "dataset";
	private static final String FIELD_PIPELINE = "pipeline";
	private static final String FIELD_PIPELINE_FEATURE = FIELD_PIPELINE + "_feature";
	private static final String FIELD_PIPELINE_REGRESSOR = FIELD_PIPELINE + "_regressor";
	private static final String FIELD_PERFORMANCE = "performance";
	private static final String FIELD_PERFORMANCES = "performances";
	private static final String FIELD_NUMBER_USAGE = "number_usage";
	private static final String FIELD_EXCEPTION = "exception";
	private static final String FIELD_AVERAGE_RUNTIME = "average_runtime";
	private static final String FIELD_RUNTIMES = "runtimes";
	private static final String FIELD_GENERATION = "generation";
	private static final String FIELD_TIMESTAMP_FOUND = "timestamp_found";

	private static final String FIELD_TYPE_LONGTEXT = "LONGTEXT";
	private static final String FIELD_TYPE_VARCHAR_255 = "VARCHAR(255)";

	private final static String TABLE_GGP_REGRESSION_RESULT_FOUND = "_ggp";
	private final static String TABLE_REGRESSOR_EVALUATED = "_pipelines";
	private final static String TABLE_FEATURE_EXTRACTOR_EVALUATED = "_features";
	private final static String TABLE_BEST_PIPELINE_FOUND = "_best";

	private final IDatabaseAdapter adapter;
	private final String dbName;
	private final String tableName;
	private final int experimentId;
	private final String executor;
	private final SimpleDateFormat format;

	public DatabaseWritingEventListener(final ExperimentConfiguration experimentConfiguration, final IDatabaseConfig dbConfig, final String executor) throws SQLException {
		this.adapter = DatabaseAdapterFactory.get(dbConfig);
		this.experimentId = experimentConfiguration.getExperimentId();
		this.executor = executor;
		this.dbName = dbConfig.getDBDatabaseName();
		this.tableName = dbConfig.getDBTableName();
		this.format = new SimpleDateFormat("YYYY-MM-dd HH:mm:ss");

		this.createGGPRegressionResultFoundEventTable();
		this.createRegressorEvaluatedEventTable();
		this.createFeatureExtractorEvaluatedEventTable();
		this.createUpdatingBestPipelineEventTable();
	}

	private void createGGPRegressionResultFoundEventTable() throws SQLException {
		Collection<String> fieldNames = new ArrayList<>();
		Map<String, String> fieldTypes = new HashMap<>();
		fieldNames.add(FIELD_PIPELINE);
		fieldNames.add(FIELD_PERFORMANCE);
		fieldNames.add(FIELD_TIMESTAMP_FOUND);

		fieldTypes.put(FIELD_PIPELINE, FIELD_TYPE_LONGTEXT);
		fieldTypes.put(FIELD_PERFORMANCE, FIELD_TYPE_VARCHAR_255);
		fieldTypes.put(FIELD_TIMESTAMP_FOUND, FIELD_TYPE_VARCHAR_255);

		this.createResultsTableIfNecessary(TABLE_GGP_REGRESSION_RESULT_FOUND, fieldNames, fieldTypes);
	}

	@Subscribe
	public void receiveGGPRegressionResultFoundEvent(final GGPRegressionResultFoundEvent event) throws SQLException {
		Map<String, Object> map = new HashMap<>();
		map.put(FIELD_PIPELINE, event.getRegressionGGPSolution().getConstructionInstruction());
		map.put(FIELD_PERFORMANCE, event.getRegressionGGPSolution().getPerformance());
		map.put(FIELD_TIMESTAMP_FOUND, this.format.format(new Date(event.getCreationTime())));
		this.insertEntry(TABLE_GGP_REGRESSION_RESULT_FOUND, map);
	}

	private void createRegressorEvaluatedEventTable() throws SQLException {
		Collection<String> fieldNames = new ArrayList<>();
		Map<String, String> fieldTypes = new HashMap<>();
		fieldNames.add(FIELD_DATASET);
		fieldNames.add(FIELD_PIPELINE_FEATURE);
		fieldNames.add(FIELD_PIPELINE_REGRESSOR);
		fieldNames.add(FIELD_PERFORMANCE);
		fieldNames.add(FIELD_AVERAGE_RUNTIME);
		fieldNames.add(FIELD_RUNTIMES);
		fieldNames.add(FIELD_EXCEPTION);
		fieldNames.add(FIELD_TIMESTAMP_FOUND);
		fieldNames.add(FIELD_GENERATION);

		fieldTypes.put(FIELD_DATASET, FIELD_TYPE_VARCHAR_255);
		fieldTypes.put(FIELD_PIPELINE_FEATURE, FIELD_TYPE_LONGTEXT);
		fieldTypes.put(FIELD_PIPELINE_REGRESSOR, FIELD_TYPE_LONGTEXT);
		fieldTypes.put(FIELD_PERFORMANCE, FIELD_TYPE_VARCHAR_255);
		fieldTypes.put(FIELD_AVERAGE_RUNTIME, FIELD_TYPE_VARCHAR_255);
		fieldTypes.put(FIELD_RUNTIMES, FIELD_TYPE_LONGTEXT);
		fieldTypes.put(FIELD_EXCEPTION, FIELD_TYPE_LONGTEXT);
		fieldTypes.put(FIELD_TIMESTAMP_FOUND, FIELD_TYPE_VARCHAR_255);
		fieldTypes.put(FIELD_GENERATION, FIELD_TYPE_VARCHAR_255);

		this.createResultsTableIfNecessary(TABLE_REGRESSOR_EVALUATED, fieldNames, fieldTypes);
	}

	@Subscribe
	public void receiveRegressorEvaluatedEvent(final RegressorEvaluatedEvent event) throws SQLException {
		Map<String, Object> map = new HashMap<>();
		map.put(FIELD_DATASET, event.getDatasetName());
		map.put(FIELD_PIPELINE_FEATURE, event.getFeatureExtractorConstructionString());
		map.put(FIELD_PIPELINE_REGRESSOR, event.getRegressorConstructionString());
		map.put(FIELD_PERFORMANCE, event.getPerformance());
		map.put(FIELD_AVERAGE_RUNTIME, event.getAverageRuntime());
		map.put(FIELD_RUNTIMES, event.getRuntimesInSeconds().toString());
		map.put(FIELD_EXCEPTION, event.getException());
		map.put(FIELD_TIMESTAMP_FOUND, this.format.format(new Date(event.getCreationTime())));
		map.put(FIELD_GENERATION, event.getGeneration());
		this.insertEntry(TABLE_REGRESSOR_EVALUATED, map);
	}

	private void createFeatureExtractorEvaluatedEventTable() throws SQLException {
		Collection<String> fieldNames = new ArrayList<>();
		Map<String, String> fieldTypes = new HashMap<>();
		fieldNames.add(FIELD_DATASET);
		fieldNames.add(FIELD_PIPELINE);
		fieldNames.add(FIELD_PERFORMANCE);
		fieldNames.add(FIELD_PERFORMANCES);
		fieldNames.add(FIELD_NUMBER_USAGE);
		fieldNames.add(FIELD_TIMESTAMP_FOUND);

		fieldTypes.put(FIELD_DATASET, FIELD_TYPE_VARCHAR_255);
		fieldTypes.put(FIELD_PIPELINE, FIELD_TYPE_LONGTEXT);
		fieldTypes.put(FIELD_PERFORMANCE, FIELD_TYPE_VARCHAR_255);
		fieldTypes.put(FIELD_PERFORMANCES, FIELD_TYPE_LONGTEXT);
		fieldTypes.put(FIELD_NUMBER_USAGE, FIELD_TYPE_VARCHAR_255);
		fieldTypes.put(FIELD_TIMESTAMP_FOUND, FIELD_TYPE_VARCHAR_255);
		this.createResultsTableIfNecessary(TABLE_FEATURE_EXTRACTOR_EVALUATED, fieldNames, fieldTypes);
	}

	@Subscribe
	public void receiveFeatureExtractorEvaluatedEvent(final FeatureExtractorEvaluatedEvent event) throws SQLException {
		Map<String, Object> map = new HashMap<>();
		map.put(FIELD_DATASET, event.getDatasetName());
		map.put(FIELD_PIPELINE, event.getPythonConstructionString());
		map.put(FIELD_PERFORMANCE, event.getAggregatedScore());
		map.put(FIELD_PERFORMANCES, event.getPerformancesOfIncludingRegressors().toString());
		map.put(FIELD_NUMBER_USAGE, event.getNumberOfUsage());
		map.put(FIELD_TIMESTAMP_FOUND, this.format.format(new Date(event.getCreationTime())));
		this.insertEntry(TABLE_FEATURE_EXTRACTOR_EVALUATED, map);
	}

	private void createUpdatingBestPipelineEventTable() throws SQLException {
		Collection<String> fieldNames = new ArrayList<>();
		Map<String, String> fieldTypes = new HashMap<>();
		fieldNames.add(FIELD_PERFORMANCE);
		fieldNames.add(FIELD_TIMESTAMP_FOUND);
		fieldNames.add(FIELD_PIPELINE);

		fieldTypes.put(FIELD_PIPELINE, FIELD_TYPE_LONGTEXT);
		fieldTypes.put(FIELD_PERFORMANCE, FIELD_TYPE_VARCHAR_255);
		fieldTypes.put(FIELD_TIMESTAMP_FOUND, FIELD_TYPE_VARCHAR_255);
		this.createResultsTableIfNecessary(TABLE_BEST_PIPELINE_FOUND, fieldNames, fieldTypes);

	}

	@Subscribe
	public void receiveUpdatingBestPipelineEvent(final UpdatingBestPipelineEvent event) throws SQLException {
		Map<String, Object> map = new HashMap<>();
		map.put(FIELD_PIPELINE, event.getConstructionString());
		map.put(FIELD_PERFORMANCE, event.getPerformance());
		map.put(FIELD_TIMESTAMP_FOUND, this.format.format(new Date(event.getCreationTime())));
		this.insertEntry(TABLE_BEST_PIPELINE_FOUND, map);
	}

	// NEW

	@SuppressWarnings("unchecked")
	@Subscribe
	public void rcvClassifierFoundEvent(final ClassifierFoundEvent event) throws SQLException {
		ScikitLearnWrapper<SingleTargetRegressionPrediction, SingleTargetRegressionPredictionBatch> learner = null;
		if (event.getSolutionCandidate() instanceof ScikitLearnWrapper) {
			learner = (ScikitLearnWrapper) event.getSolutionCandidate();
			// this.logCandidateEvaluation("success", learner.toString(), event.getInSampleError() + "", event.getTimeToEvaluate() + "ms");

			this.receiveRegressorEvaluatedEvent(new RegressorEvaluatedEvent(learner.toString(), "", "", event.getInSampleError(), Arrays.asList((long) event.getTimeToEvaluate()), ""));
		}
		// TABLE_REGRESSOR_EVALUATED
	}

	@Subscribe
	public void rcvTrainTestSplitEvaluationFailedEvent(final TrainTestSplitEvaluationFailedEvent<ILabeledInstance, ILabeledDataset<? extends ILabeledInstance>> event) throws SQLException {
		ScikitLearnWrapper<SingleTargetRegressionPrediction, SingleTargetRegressionPredictionBatch> learner = null;
		if (event.getLearner() instanceof ScikitLearnWrapper) {
			learner = (ScikitLearnWrapper) event.getLearner();
		}
		if (learner != null) {
			ILearnerRunReport report = event.getFirstReport();
			// String status = "unknown";
			// if (event.getReport().getException() instanceof LearnerExecutionInterruptedException) {
			// status = "timeout";
			// } else if (event.getReport().getException() instanceof LearnerExecutionFailedException) {
			// if (ExceptionUtils.getStackTrace(event.getReport().getException()).contains("NoSuchFileException")) {
			// status = "timeout (python)";
			// } else if (ExceptionUtils.getStackTrace(event.getReport().getException()).contains("MemoryError: Unable to allocate")) {
			// status = "memory full";
			// } else {
			// status = "crashed";
			// }
			// }
			// String exceptionStackTrace = ExceptionUtils.getStackTrace(report.getException());
			//
			// this.logCandidateEvaluation(status, learner.toString(), exceptionStackTrace, candidateRuntime + "ms");
			String exceptionMessage = "";
			if (report.getException() != null) {
				exceptionMessage = ExceptionUtils.getStackTrace(report.getException());
			}
			this.receiveRegressorEvaluatedEvent(new RegressorEvaluatedEvent(learner.toString(), "", "", exceptionMessage, ""));
		}
	}

	// END

	private void createResultsTableIfNecessary(final String suffix, final Collection<String> fieldNames, final Map<String, String> fieldTypes) throws SQLException {
		List<IKVStore> resultSet = this.adapter.getResultsOfQuery("SHOW TABLES");
		boolean resultTableAlreadyExists = resultSet.stream().anyMatch(kvStore -> kvStore.getAsString("Tables_in_" + this.dbName).equals(this.tableName + suffix));
		if (!resultTableAlreadyExists) {
			fieldNames.add(FIELD_EXPERIMENT_ID);
			fieldNames.add(FIELD_EXECUTOR);
			fieldNames.add(FIELD_THREAD);

			fieldTypes.put(FIELD_PRIMARY_KEY, FIELD_TYPE_INT);
			fieldTypes.put(FIELD_EXPERIMENT_ID, FIELD_TYPE_VARCHAR_255);
			fieldTypes.put(FIELD_EXECUTOR, FIELD_TYPE_VARCHAR_255);
			fieldTypes.put(FIELD_THREAD, FIELD_TYPE_VARCHAR_255);

			this.adapter.createTable(this.tableName + suffix, FIELD_PRIMARY_KEY, fieldNames, fieldTypes, null);
		}
	}

	private void insertEntry(final String suffix, final Map<String, Object> map) throws SQLException {
		map.put(FIELD_EXPERIMENT_ID, this.experimentId);
		map.put(FIELD_EXECUTOR, this.executor);
		map.put(FIELD_THREAD, Thread.currentThread().getName());
		new KVStore(map);
		this.adapter.insert(this.tableName + suffix, map);
	}

}
