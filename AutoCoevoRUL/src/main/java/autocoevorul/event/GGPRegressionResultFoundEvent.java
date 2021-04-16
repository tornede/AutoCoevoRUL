package autocoevorul.event;

import autocoevorul.regression.RegressionGGPSolution;

public class GGPRegressionResultFoundEvent extends AbstractEvent {

	private RegressionGGPSolution regressionGGPSolution;

	public GGPRegressionResultFoundEvent(final RegressionGGPSolution regressionGGPSolution) {
		super();
		this.regressionGGPSolution = regressionGGPSolution;
	}

	public RegressionGGPSolution getRegressionGGPSolution() {
		return this.regressionGGPSolution;
	}

}
