package autocoevorul.experiment;

import java.lang.reflect.Method;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.aeonbits.owner.Config.Sources;
import org.aeonbits.owner.Converter;
import org.api4.java.algorithm.Timeout;

import ai.libs.jaicore.experiments.IExperimentSetConfig;
import ai.libs.jaicore.ml.regression.loss.ERulPerformanceMeasure;
import autocoevorul.regression.featurerating.EFeatureRater;
import autocoevorul.regression.featurerating.IFeatureRater;

@Sources({ "file:./conf/experiments/experiments.cnf" })
public interface ICoevolutionConfig extends IExperimentSetConfig {

	@Key("experimentId")
	@DefaultValue("1")
	public int getExperimentId();

	@Key("dataPath")
	@DefaultValue("../../Data/")
	public String getDataPath();

	@Key("datasetName")
	@DefaultValue("CMAPSS/FD001_train.arff")
	public String getDatasetName();

	@Key("seed")
	@DefaultValue("42")
	public int getSeed();

	@Key("numberOfFolds")
	@DefaultValue("5")
	public int getNumberOfFolds();

	@Key("featureSearchspace")
	@DefaultValue("searchspace/timeseries/timeseries_feature_extraction.json")
	public String getFeatureSearchspace();

	@Key("featureRequiredInterface")
	@DefaultValue("TimeSeriesFeatureGenerator")
	public String getFeatureRequiredInterfacee();

	@Key("featureMainComponentNames")
	@DefaultValue("python_connection.feature_generation.tsfresh_feature_generator.FCParametersDictionary, pyts.transformation.BagOfPatterns, pyts.transformation.BOSS, pyts.transformation.ROCKET, pyts.transformation.ShapeletTransform, python_connection.feature_generation.ultra_fast_shapelets_feature_generator.UltraFastShapeletsFeatureExtractor, pyts.transformation.WEASEL")
	@Separator(",")
	public List<String> getFeatureMainComponentNames();

	@Key("featureMainComponentNamesWithoutActivationBit")
	@DefaultValue("python_connection.feature_generation.tsfresh_feature_generator.FCParametersDictionary")
	@Separator(",")
	public List<String> getFeatureMainComponentNamesWithoutActivationBit();

	@Key("featureCandidateTimeout")
	@DefaultValue("5:MINUTES")
	@ConverterClass(TimeoutConverter.class)
	public Timeout getFeatureCandidateTimeout();

	@Key("featurePopulationSize")
	@DefaultValue("10")
	public int getFeaturePopulationSize();

	@Key("totalTimeout")
	@DefaultValue("20:MINUTES")
	@ConverterClass(TimeoutConverter.class)
	public Timeout getTotalTimeout();

	@Key("regressionSearchpace")
	@DefaultValue("searchspace/regression/index.json")
	public String getRegressionSearchpace();

	@Key("featureObjectiveMeasure")
	@DefaultValue("AVERAGE")
	@ConverterClass(EFeatureRaterConverter.class)
	public IFeatureRater getFeatureObjectiveMeasure();

	@Key("regressionRequiredInterface")
	@DefaultValue("AbstractRegressor")
	public String getRegressionRequiredInterface();

	@Key("regressionPopulationSize")
	@DefaultValue("10")
	public int getRegressionPopulationSize();

	@Key("regressionCandidateTimeout")
	@DefaultValue("5:MINUTES")
	@ConverterClass(TimeoutConverter.class)
	public Timeout getRegressionCandidateTimeout();

	@Key("rulSearchspace")
	@DefaultValue("searchspace/rul.json")
	public String getRulSearchSpace();

	@Key("rulRulRootComponentName")
	@DefaultValue("sklearn.pipeline.make_pipeline")
	public String getRulRulRootComponentName();

	@Key("performanceMeasure")
	@DefaultValue("ASYMMETRIC_LOSS")
	@ConverterClass(ERulPerformanceMeasureConverter.class)
	public ERulPerformanceMeasure getPerformanceMeasure();

	public class TimeoutConverter implements Converter<Timeout> {
		@Override
		public Timeout convert(final Method targetMethod, final String text) {
			final String[] split = text.split(":", -1);
			final Integer amount = Integer.parseInt(split[0]);
			final TimeUnit unit = TimeUnit.valueOf(split[1]);
			return new Timeout(amount, unit);
		}
	}

	public class EFeatureRaterConverter implements Converter<IFeatureRater> {
		@Override
		public IFeatureRater convert(final Method targetMethod, final String text) {
			return EFeatureRater.valueOf(text);
		}
	}

	public class ERulPerformanceMeasureConverter implements Converter<ERulPerformanceMeasure> {
		@Override
		public ERulPerformanceMeasure convert(final Method targetMethod, final String text) {
			return ERulPerformanceMeasure.valueOf(text);
		}
	}

}
