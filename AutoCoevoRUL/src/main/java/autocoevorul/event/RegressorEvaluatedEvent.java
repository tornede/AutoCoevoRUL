package autocoevorul.event;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class RegressorEvaluatedEvent extends AbstractEvent {

	private String regressorConstructionString;
	private String datasetName;
	private String featureExtractorConstructionString;
	private double performance;

	private String exception = null;
	private List<Long> runtimes;
	private String generation;

	public RegressorEvaluatedEvent(final String regressorConstructionString, final String featureExtractorConstructionString, final String datasetName, final double performance,
			final String exception, final List<Long> runtimes, final String generation) {
		super();
		this.regressorConstructionString = regressorConstructionString;
		this.featureExtractorConstructionString = featureExtractorConstructionString;
		this.datasetName = datasetName;
		this.performance = performance;
		this.exception = exception;
		this.runtimes = runtimes;
		this.generation = generation;
	}

	public RegressorEvaluatedEvent(final String regressorConstructionString, final String featureExtractorConstructionString, final String datasetName, final double score, final List<Long> runtimes,
			final String generation) {
		this(regressorConstructionString, featureExtractorConstructionString, datasetName, score, null, runtimes, generation);
	}

	public RegressorEvaluatedEvent(final String regressorConstructionString, final String featureExtractorConstructionString, final String datasetName, final String exception,
			final String generation) {
		this(regressorConstructionString, featureExtractorConstructionString, datasetName, -1, exception, new ArrayList<>(), generation);
	}

	public String getRegressorConstructionString() {
		return this.regressorConstructionString;
	}

	public String getDatasetName() {
		return this.datasetName;
	}

	public double getPerformance() {
		return this.performance;
	}

	public boolean isError() {
		return this.exception != null;
	}

	public String getGeneration() {
		return this.generation;
	}

	public String getFeatureExtractorConstructionString() {
		return this.featureExtractorConstructionString;
	}

	public List<Double> getRuntimesInSeconds() {
		return this.runtimes.stream().map(l -> l / 1_000.0).collect(Collectors.toList());
	}

	public double getAverageRuntime() {
		if (this.runtimes.isEmpty()) {
			return -1;
		}
		return this.runtimes.stream().mapToDouble(l -> l / 1_000.0).average().getAsDouble();
	}

	public String getException() {
		if (this.exception == null) {
			return "";
		}
		return this.exception;
	}

}
