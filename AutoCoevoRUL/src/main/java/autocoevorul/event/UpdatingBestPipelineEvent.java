package autocoevorul.event;

public class UpdatingBestPipelineEvent extends AbstractEvent {

	private String constructionString;
	private double performance;

	public UpdatingBestPipelineEvent(final String constructionString, final double performance) {
		super();
		this.constructionString = constructionString;
		this.performance = performance;
	}

	public String getConstructionString() {
		return this.constructionString;
	}

	public double getPerformance() {
		return this.performance;
	}

}
