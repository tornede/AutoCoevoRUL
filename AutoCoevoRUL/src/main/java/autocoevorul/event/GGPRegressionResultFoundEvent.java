package autocoevorul.event;

import autocoevorul.regression.GgpRegressionSolution;

public class GGPRegressionResultFoundEvent extends AbstractEvent {

	private GgpRegressionSolution regressionGGPSolution;

	public GGPRegressionResultFoundEvent(final GgpRegressionSolution regressionGGPSolution) {
		super();
		this.regressionGGPSolution = regressionGGPSolution;
	}

	public GgpRegressionSolution getRegressionGGPSolution() {
		return this.regressionGGPSolution;
	}

}
