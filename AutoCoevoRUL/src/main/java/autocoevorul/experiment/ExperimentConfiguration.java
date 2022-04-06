package autocoevorul.experiment;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;
import java.util.concurrent.TimeUnit;

import org.aeonbits.owner.ConfigCache;
import org.aeonbits.owner.ConfigFactory;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.algorithm.Timeout;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.ml.core.dataset.serialization.ArffDatasetAdapter;
import ai.libs.jaicore.ml.hpo.ggp.IGrammarBasedGeneticProgrammingConfig;
import ai.libs.jaicore.ml.regression.loss.ERulPerformanceMeasure;
import ai.libs.jaicore.ml.scikitwrapper.IScikitLearnWrapperConfig;
import autocoevorul.regression.featurerating.EFeatureRater;
import autocoevorul.regression.featurerating.IFeatureRater;

public class ExperimentConfiguration {

	private static final Logger LOGGER = LoggerFactory.getLogger("experiment");

	private static final String CONFIG_FILE_PATH = "conf/experiments/experiments.cnf";

	private String experimentConfigFilePath;
	private int experimentId;
	private String datasetName;
	private String trainingDataPath;
	private String evaluationDataPath;
	private ILabeledDataset<ILabeledInstance> trainingData;
	private ILabeledDataset<ILabeledInstance> evaluationData;

	private long seed;
	private int numberOfFolds;
	private int numberOfCPUs;

	private String featureSearchspace;
	private String featureRequiredInterface;
	private List<String> featureMainComponentNames;
	private List<String> featureMainComponentNamesWithoutActivationBit;
	private int featurePopulationSize;
	private Timeout totalTimeout;
	private Timeout featureGenerationTimeout;
	private Timeout featureCandidateTimeout;
	private IFeatureRater featureObjectiveMeasure;

	private String regressionSearchpace;
	private String regressionRequiredInterface;
	private Timeout regressionGenerationTimeout;
	private Timeout regressionCandidateTimeout;

	private String rulSearchSpace;
	private String rulRulRootComponentName;
	private Timeout rulTimeout;

	private long deadline;

	private ERulPerformanceMeasure performanceMeasure;

	private IScikitLearnWrapperConfig scikitLearnWrapperConfig;

	public ExperimentConfiguration(final String experimentConfigFilePath, final ICoevolutionConfig experimentSetConfig) {
		this.experimentConfigFilePath = experimentConfigFilePath;
		this.experimentId = experimentSetConfig.getExperimentId();
		this.datasetName = experimentSetConfig.getDatasetName();
		this.trainingDataPath = experimentSetConfig.getDataPath() + experimentSetConfig.getDatasetName();
		this.evaluationDataPath = experimentSetConfig.getDataPath() + experimentSetConfig.getDatasetName().replace("train", "test");
		try {
			this.seed = experimentSetConfig.getSeed();
		} catch (final Exception e) {
			LOGGER.warn("Could not read seed from config file.");
		}
		this.numberOfFolds = experimentSetConfig.getNumberOfFolds();
		this.numberOfCPUs = experimentSetConfig.getNumberOfCPUs();

		this.featureSearchspace = experimentSetConfig.getFeatureSearchspace();
		this.featureRequiredInterface = experimentSetConfig.getFeatureRequiredInterfacee();
		this.featureMainComponentNames = experimentSetConfig.getFeatureMainComponentNames();
		this.featureMainComponentNamesWithoutActivationBit = experimentSetConfig.getFeatureMainComponentNamesWithoutActivationBit();
		this.featurePopulationSize = experimentSetConfig.getFeaturePopulationSize();
		this.totalTimeout = experimentSetConfig.getTotalTimeout();
		this.featureCandidateTimeout = experimentSetConfig.getFeatureCandidateTimeout();
		try {
			this.featureObjectiveMeasure = experimentSetConfig.getFeatureObjectiveMeasure();
		} catch (final Exception e) {
			LOGGER.warn("Could not read feature objective measure from config file.");
		}

		this.regressionSearchpace = experimentSetConfig.getRegressionSearchpace();
		this.regressionRequiredInterface = experimentSetConfig.getRegressionRequiredInterface();
		this.regressionCandidateTimeout = experimentSetConfig.getRegressionCandidateTimeout();

		this.rulSearchSpace = experimentSetConfig.getRulSearchSpace();
		this.rulRulRootComponentName = experimentSetConfig.getRulRulRootComponentName();

		this.performanceMeasure = experimentSetConfig.getPerformanceMeasure();

		this.setup();
	}

	public ExperimentConfiguration(final String experimentConfigFilePath) {
		this(experimentConfigFilePath, (ICoevolutionConfig) ConfigFactory.create(ICoevolutionConfig.class).loadPropertiesFromFile(new File(experimentConfigFilePath)));
	}

	public ExperimentConfiguration() {
		this(CONFIG_FILE_PATH);
	}

	public ExperimentConfiguration(final String experimentConfigFilePath, final ICoevolutionConfig experimentSetConfig, final ExperimentDBEntry experimentEntry) throws ExperimentEvaluationFailedException {
		this(experimentConfigFilePath, experimentSetConfig);

		final Map<String, String> keyFields = experimentEntry.getExperiment().getValuesOfKeyFields();

		this.experimentId = experimentEntry.getId();
		this.datasetName = keyFields.get("datasetName");
		this.trainingDataPath = experimentSetConfig.getDataPath() + keyFields.get("datasetName");
		this.evaluationDataPath = experimentSetConfig.getDataPath() + keyFields.get("datasetName").replace("train", "test");

		this.featureObjectiveMeasure = EFeatureRater.valueOf(keyFields.get("featureObjectiveMeasure"));
		this.performanceMeasure = ERulPerformanceMeasure.valueOf(keyFields.get("internal_performance_measure"));

		this.seed = Long.parseLong(keyFields.get("seed"));

		this.setup();
	}

	private void setup() {
		this.datasetName = this.datasetName.replace("/", "_").replace("_train", "").replace(".arff", "");
		if (this.featureMainComponentNamesWithoutActivationBit == null) {
			this.featureMainComponentNamesWithoutActivationBit = new ArrayList<>();
			System.out.println(this.featureMainComponentNamesWithoutActivationBit);
		}

		this.featureGenerationTimeout = new Timeout(this.featureCandidateTimeout.seconds() * this.featurePopulationSize + this.featurePopulationSize, TimeUnit.SECONDS);
		this.regressionGenerationTimeout = new Timeout(this.regressionCandidateTimeout.seconds() * this.getRegressionGGPConfig().getPopulationSize() + this.getRegressionGGPConfig().getPopulationSize(), TimeUnit.SECONDS);
		this.rulTimeout = new Timeout(this.featureCandidateTimeout.milliseconds() + this.regressionCandidateTimeout.milliseconds(), TimeUnit.MILLISECONDS);

		this.scikitLearnWrapperConfig = ConfigCache.getOrCreate(IScikitLearnWrapperConfig.class);
		this.scikitLearnWrapperConfig.setProperty("sklearn.wrapper.temp.folder", "tmp/tmp" + this.experimentId + "/");

		this.deadline = System.currentTimeMillis() + this.totalTimeout.milliseconds();
	}

	public String getExperimentConfigFilePath() {
		return this.experimentConfigFilePath;
	}

	public int getExperimentId() {
		return this.experimentId;
	}

	public String getDatasetName() {
		return this.datasetName;
	}

	public String getTrainingDataPath() {
		return this.trainingDataPath;
	}

	public ILabeledDataset<ILabeledInstance> getTrainingData() throws ExperimentEvaluationFailedException {
		if (this.trainingData == null) {
			this.trainingData = this.readDataFile(this.trainingDataPath);
		}
		return this.trainingData;
	}

	public ILabeledDataset<ILabeledInstance> getEvaluationData() throws ExperimentEvaluationFailedException {
		if (this.evaluationData == null) {
			this.evaluationData = this.readDataFile(this.evaluationDataPath);
		}
		return this.evaluationData;
	}

	public ILabeledDataset<ILabeledInstance> readDataFile(final String filePath) throws ExperimentEvaluationFailedException {
		final long start = System.currentTimeMillis();
		ILabeledDataset<ILabeledInstance> dataset;
		try {
			dataset = new ArffDatasetAdapter().readDataset(new File(filePath));
		} catch (final Exception e) {
			throw new ExperimentEvaluationFailedException("Could not deserialize dataset: " + new File(filePath).getAbsolutePath(), e);
		}
		LOGGER.info("Data {} read. Time to create dataset object was {}ms", filePath, System.currentTimeMillis() - start);
		return dataset;

	}

	public Map<String, String> getTemplateVariables() {
		final HashMap<String, String> templateVariables = new HashMap<>();
		try {
			templateVariables.put("number_of_attributes_minus_one", "" + (this.getTrainingData().getListOfAttributes().size() - 1));
		} catch (ExperimentEvaluationFailedException e) {
			throw new RuntimeException(e);
		}
		return templateVariables;
	}

	public long getSeed() {
		return this.seed;
	}

	public int getNumberOfFolds() {
		return this.numberOfFolds;
	}

	public int getNumCPUs() {
		return this.numberOfCPUs;
	}

	public String getFeatureSearchspace() {
		return this.featureSearchspace;
	}

	public String getFeatureRequiredInterface() {
		return this.featureRequiredInterface;
	}

	public List<String> getFeatureMainComponentNames() {
		return this.featureMainComponentNames;
	}

	public List<String> getFeatureMainComponentNamesWithoutActivationBit() {
		return this.featureMainComponentNamesWithoutActivationBit;
	}

	public int getFeaturePopulationSize() {
		return this.featurePopulationSize;
	}

	public Timeout getTotalTimeout() {
		return this.totalTimeout;
	}

	public Timeout getFeatureGenerationTimeout() {
		return this.featureGenerationTimeout;
	}

	public Timeout getFeatureCandidateTimeoutPerFold() {
		return new Timeout((this.featureCandidateTimeout.seconds() / this.numberOfFolds) - 1, TimeUnit.SECONDS);
	}

	public Timeout getFeatureCandidateTimeout() {
		return this.featureCandidateTimeout;
	}

	public IFeatureRater getFeatureObjectiveMeasure() {
		return this.featureObjectiveMeasure;
	}

	public String getRegressionSearchpace() {
		return this.regressionSearchpace;
	}

	public String getRegressionRequiredInterface() {
		return this.regressionRequiredInterface;
	}

	public IGrammarBasedGeneticProgrammingConfig getRegressionGGPConfig() {
		final IGrammarBasedGeneticProgrammingConfig config = (IGrammarBasedGeneticProgrammingConfig) ConfigFactory.create(IGrammarBasedGeneticProgrammingConfig.class).loadPropertiesFromFile(new File(this.experimentConfigFilePath));
		config.setProperty("timeout", "" + (Math.max(1, this.deadline - System.currentTimeMillis())));
		config.setProperty("cpus", "" + (this.numberOfCPUs));
		return config;
	}

	public Timeout getRegressionGenerationTimeout() {
		return this.regressionGenerationTimeout;
	}

	public Timeout getRegressionCandidateTimeout() {
		return this.regressionCandidateTimeout;
	}

	public String getRulSearchSpace() {
		return this.rulSearchSpace;
	}

	public String getRulRootComponentName() {
		return this.rulRulRootComponentName;
	}

	public Timeout getRulTimeout() {
		return this.rulTimeout;
	}

	public ERulPerformanceMeasure getPerformanceMeasure() {
		return this.performanceMeasure;
	}

	public IScikitLearnWrapperConfig getScikitLearnWrapperConfig() {
		return this.scikitLearnWrapperConfig;
	}

	@Override
	public String toString() {
		final StringJoiner sj = new StringJoiner("\n\t");
		sj.add("ExperimentConfiguration:");
		sj.add("experimentId=" + this.experimentId);
		sj.add("trainingDataPath=" + this.trainingDataPath);
		sj.add("evaluationDataPath=" + this.evaluationDataPath);
		sj.add("seed=" + this.seed);
		sj.add("numberOfFolds=" + this.numberOfFolds);
		sj.add("numberOfCPUs=" + this.numberOfCPUs);
		sj.add("featureSearchspace=" + this.featureSearchspace);
		sj.add("featureRequiredInterface=" + this.featureRequiredInterface);
		sj.add("featureMainComponentNames=" + this.featureMainComponentNames);
		sj.add("featureMainComponentNamesWithoutActivationBit=" + this.featureMainComponentNamesWithoutActivationBit);
		sj.add("featurePopulationSize=" + this.featurePopulationSize);
		sj.add("totalTimeout=" + this.totalTimeout);
		sj.add("featureGenerationTimeout=" + this.featureGenerationTimeout);
		sj.add("featureCandidateTimeout=" + this.featureCandidateTimeout);
		sj.add("regressionSearchpace=" + this.regressionSearchpace);
		sj.add("regressionRequiredInterface=" + this.regressionRequiredInterface);
		sj.add("regressionGGPConfig=" + this.getRegressionGGPConfig());
		sj.add("regressionGenerationTimeout=" + this.regressionGenerationTimeout);
		sj.add("regressionCandidateTimeout=" + this.regressionCandidateTimeout);
		sj.add("performanceMeasure=" + this.performanceMeasure);
		sj.add("featureObjectiveMeasure=" + this.featureObjectiveMeasure);
		return sj.toString();
	}

}
