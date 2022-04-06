package autocoevorul.featurerextraction;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledInstance;
import org.api4.java.ai.ml.core.evaluation.IPrediction;
import org.api4.java.ai.ml.core.evaluation.IPredictionBatch;
import org.api4.java.ai.ml.core.exception.PredictionException;
import org.api4.java.ai.ml.core.exception.TrainingException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.basic.FileUtil;
import ai.libs.jaicore.ml.core.EScikitLearnProblemType;
import ai.libs.jaicore.ml.scikitwrapper.AScikitLearnWrapper;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnWrapperCommandBuilder;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnWrapperExecutionFailedException;

public class ML4PdMTimeSeriesFeatureEngineeringWrapper extends AScikitLearnWrapper<IPrediction, IPredictionBatch> {

	protected static Logger LOGGER = LoggerFactory.getLogger(ML4PdMTimeSeriesFeatureEngineeringWrapper.class);

	private final String ARFF_FILE_ENDING = ".arff";
	private final String PDMFF_FILE_ENDING = ".pdmff";

	public ML4PdMTimeSeriesFeatureEngineeringWrapper(final String pipeline, final String imports) throws IOException, InterruptedException {
		super(EScikitLearnProblemType.TIME_SERIES_FEATURE_ENGINEERING, pipeline, imports);
		this.setPythonTemplate("src/main/resources/ml4pdm.py");
	}

	@Override
	protected boolean doLabelsFitToProblemType(final ILabeledDataset<? extends ILabeledInstance> data) {
		return true;
	}

	@Override
	public String getDataName(final ILabeledDataset<? extends ILabeledInstance> data) {
		return data.getRelationName();
	}

	@Override
	public File getOutputFile(final String dataName) {
		String datasetFileName = dataName;
		if (dataName.endsWith(this.ARFF_FILE_ENDING)) {
			datasetFileName = datasetFileName.split(this.ARFF_FILE_ENDING)[0];
		}
		if (!dataName.endsWith(this.PDMFF_FILE_ENDING)) {
			datasetFileName += this.PDMFF_FILE_ENDING;
		}
		return new File(this.scikitLearnWrapperConfig.getTempFolder(), this.configurationUID + "_" + datasetFileName);
	}

	@Override
	protected ScikitLearnWrapperCommandBuilder getCommandBuilder() {
		ScikitLearnTimeSeriesFeatureEngineeringWrapperCommandBuilder commandBuilder = new ScikitLearnTimeSeriesFeatureEngineeringWrapperCommandBuilder(this.problemType.getScikitLearnCommandLineFlag(), this.getSKLearnScriptFile());
		return super.getCommandBuilder(commandBuilder);
	}

	@Override
	protected ScikitLearnWrapperCommandBuilder constructCommandLineParametersForFitMode(final File modelFile, final File trainingDataFile) {
		return super.constructCommandLineParametersForFitMode(modelFile, trainingDataFile).withFitOutputFile(this.getOutputFile(trainingDataFile.getName()));
	}

	@Override
	protected ScikitLearnWrapperCommandBuilder constructCommandLineParametersForFitAndPredictMode(final File trainingDataFile, final File testingDataFile, final File testingOutputFile) {
		return super.constructCommandLineParametersForFitAndPredictMode(trainingDataFile, testingDataFile, testingOutputFile).withFitOutputFile(this.getOutputFile(trainingDataFile.getName()));
	}

	@Override
	public IPredictionBatch fitAndPredict(final File trainingDataFile, final String trainingDataName, final File testingDataFile, final String testingDataName) throws TrainingException, PredictionException, InterruptedException {
		try {
			File trainingOutputFile = this.getOutputFile(trainingDataName);
			File testingOutputFile = this.getOutputFile(testingDataName);

			if (!trainingOutputFile.exists() && !testingOutputFile.exists()) {
				String[] fitAndPredictCommand = this.constructCommandLineParametersForFitAndPredictMode(trainingDataFile, testingDataFile, testingOutputFile).toCommandArray();
				if (this.logger.isDebugEnabled()) {
					this.logger.debug("{} run fitAndPredict mode {}", Thread.currentThread().getName(), Arrays.toString(fitAndPredictCommand));
				}
				this.runProcess(fitAndPredictCommand);
			}

			this.handleOutput(trainingOutputFile);
			this.handleOutput(testingOutputFile);
			
		} catch (ScikitLearnWrapperExecutionFailedException e) {
			throw new TrainingException("Could not run scikit-learn model.", e);
		}
		return null;
	}

	@Override
	protected IPredictionBatch handleOutput(final File outputFile) throws TrainingException {
		if (!outputFile.exists()) {
			FileUtil.touch(outputFile.getAbsolutePath());
			throw new TrainingException("Executing python failed as the following file does not exist: " + outputFile);
		}
		return null;
	}

	class ScikitLearnTimeSeriesFeatureEngineeringWrapperCommandBuilder extends ScikitLearnWrapperCommandBuilder {

		protected ScikitLearnTimeSeriesFeatureEngineeringWrapperCommandBuilder(final String problemTypeFlag, final File scriptFile) {
			super(problemTypeFlag, scriptFile);
		}

		@Override
		protected void checkRequirementsTrainMode() {
			Objects.requireNonNull(this.fitDataFile);
			Objects.requireNonNull(this.modelFile);
			Objects.requireNonNull(this.fitOutputFile);
		}

		@Override
		protected void checkRequirementsTrainTestMode() {
			Objects.requireNonNull(this.fitDataFile);
			Objects.requireNonNull(this.fitOutputFile);
			Objects.requireNonNull(this.predictDataFile);
			Objects.requireNonNull(this.predictOutputFile);
		}

	}

}
