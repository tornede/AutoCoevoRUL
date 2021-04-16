package autocoevorul;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.eventbus.EventBus;

import autocoevorul.baseline.randomsearch.PipelineEvaluationReport;
import autocoevorul.event.UpdatingBestPipelineEvent;

public class SearchResult {

	private Logger LOGGER = LoggerFactory.getLogger(SearchResult.class);

	private EventBus eventBus;
	private PipelineEvaluationReport pipelineEvaluationReport;

	public SearchResult(final EventBus eventBus) {
		this.eventBus = eventBus;
		this.pipelineEvaluationReport = new PipelineEvaluationReport("", "", Double.MAX_VALUE, 0, 0);
	}

	public SearchResult(final EventBus eventBus, final PipelineEvaluationReport pipelineEvaluationReport) {
		this.pipelineEvaluationReport = pipelineEvaluationReport;
	}

	public synchronized void update(final PipelineEvaluationReport report) {
		if (!report.isFailed() && report.getPerformance() < this.pipelineEvaluationReport.getPerformance()) {
			this.pipelineEvaluationReport = report;
			this.eventBus.post(new UpdatingBestPipelineEvent(this.pipelineEvaluationReport.getConstructionInstruction(), this.pipelineEvaluationReport.getPerformance()));
			this.LOGGER.info("New incumbent found: {} {} ", this.pipelineEvaluationReport.getPerformance(), this.pipelineEvaluationReport.getConstructionInstruction());
		}
	}

	public PipelineEvaluationReport getPipelineEvaluationReport() {
		return this.pipelineEvaluationReport;
	}

}
