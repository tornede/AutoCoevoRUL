package autocoevorul.featurerextraction;

import java.io.IOException;

import org.api4.java.ai.ml.core.evaluation.IPrediction;
import org.api4.java.ai.ml.core.evaluation.IPredictionBatch;

import ai.libs.jaicore.ml.core.EScikitLearnProblemType;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnRegressionWrapper;

public class TimeseriesFeatureEngineeringScikitLearnWrapper<P extends IPrediction, B extends IPredictionBatch> extends ScikitLearnRegressionWrapper<P, B> {

	public TimeseriesFeatureEngineeringScikitLearnWrapper(final String pipeline, final String imports) throws IOException, InterruptedException {
		super(EScikitLearnProblemType.TIME_SERIES_FEATURE_ENGINEERING, pipeline, imports);
	}

}
