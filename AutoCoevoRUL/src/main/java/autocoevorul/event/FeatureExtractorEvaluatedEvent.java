package autocoevorul.event;

import java.util.List;

public class FeatureExtractorEvaluatedEvent extends AbstractEvent {

	private String pythonConstructionString;
	private String datasetName;
	private double aggregatedScore;
	private List<Double> performancesOfIncludingRegressors;
	private double numberOfUsage;

	public FeatureExtractorEvaluatedEvent(final String pythonConstructionString, final String datasetName, final double aggregatedScore, final List<Double> performancesOfIncludingRegressors, final double numberOfUsage) {
		this.pythonConstructionString = pythonConstructionString;
		this.datasetName = datasetName;
		this.aggregatedScore = aggregatedScore;
		this.performancesOfIncludingRegressors = performancesOfIncludingRegressors;
		this.numberOfUsage = numberOfUsage;
	}

	public String getPythonConstructionString() {
		return this.pythonConstructionString;
	}

	public String getDatasetName() {
		return this.datasetName;
	}

	public double getAggregatedScore() {
		return this.aggregatedScore;
	}

	public List<Double> getPerformancesOfIncludingRegressors() {
		return this.performancesOfIncludingRegressors;
	}

	public double getNumberOfUsage() {
		return this.numberOfUsage;
	}
}
