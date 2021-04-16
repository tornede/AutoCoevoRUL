package autocoevorul.baseline.randomsearch;

import ai.libs.jaicore.components.api.IComponentInstance;

public class PipelineEvaluationReport {

	private String constructionInstruction;
	private String imports;
	private Double performance;
	private long startTime;
	private long endTime;

	public PipelineEvaluationReport(final String constructionInstruction, final String imports, final double performance, final long startTime, final long endTime) {
		super();
		this.constructionInstruction = constructionInstruction;
		this.imports = imports;
		this.performance = performance;
		this.startTime = startTime;
		this.endTime = endTime;
	}

	public PipelineEvaluationReport(final IComponentInstance componentInstance, final String constructionInstruction, final String imports, final long startTime, final long endTime) {
		super();
		this.constructionInstruction = constructionInstruction;
		this.imports = imports;
		this.performance = null;
		this.startTime = startTime;
		this.endTime = endTime;
	}

	public String getConstructionInstruction() {
		return this.constructionInstruction;
	}

	public String getImports() {
		return this.imports;
	}

	public double getPerformance() {
		return this.performance.doubleValue();
	}

	public long getStartTime() {
		return this.startTime;
	}

	public long getEndTime() {
		return this.endTime;
	}

	public boolean isFailed() {
		return this.performance == null;
	}

}
