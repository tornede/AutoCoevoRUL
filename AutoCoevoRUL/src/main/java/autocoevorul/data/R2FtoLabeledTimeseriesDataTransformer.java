package autocoevorul.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.StringJoiner;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Stream;

public class R2FtoLabeledTimeseriesDataTransformer implements DatasetTransformer {

	private static final Logger LOGGER = Logger.getLogger(R2FtoLabeledTimeseriesDataTransformer.class.getName());
	private static final int MIN_REMAINING_TIMESTEPS_AFTER_FAILURE = 10;
	private static final int MAX_REMAINING_TIMESTEPS_AFTER_FAILURE = 150;

	private long seed = 4837692L;
	private Random random;
	private String relation;
	private List<String> attributeNames;
	private List<String> attributeTypes;
	private int copyFactor;
	private int timeSeriesStartingIndex;

	public R2FtoLabeledTimeseriesDataTransformer() {
		this.random = new Random(this.seed);
		this.copyFactor = 1;
	}

	public R2FtoLabeledTimeseriesDataTransformer(final int timeSeriesStartingIndex, final int copyFactor) {
		this.random = new Random(this.seed);
		this.timeSeriesStartingIndex = timeSeriesStartingIndex;
		this.copyFactor = copyFactor;
	}

	public R2FtoLabeledTimeseriesDataTransformer(final int timeSeriesStartingIndex) {
		this(timeSeriesStartingIndex, 1);
	}

	@Override
	public String transform(final String inputFileName) {
		if (!this.canRead(inputFileName)) {
			LOGGER.log(Level.SEVERE, "can't read input file: " + inputFileName);
			return null;
		}

		LOGGER.info("Transforming input file: " + inputFileName);

		Path filePath = Paths.get(inputFileName);
		List<String> instanceLines = new ArrayList<>();
		try (Stream<String> lines = Files.lines(filePath)) {
			lines.forEach(e -> instanceLines.add(e));
		} catch (IOException e) {
		}

		int dataStartingIndex = instanceLines.indexOf(Constants.ARFF_HEADER_TAG_DATA) + 1;
		this.attributeNames = new ArrayList<>();
		this.attributeTypes = new ArrayList<>();
		for (int i = 0; i < dataStartingIndex; i++) {
			if (instanceLines.get(i).startsWith(Constants.ARFF_HEADER_TAG_RELATION)) {
				this.relation = instanceLines.get(i).split(" ")[1];
			} else if (instanceLines.get(i).startsWith(Constants.ARFF_HEADER_TAG_ATTRIBUTE)) {
				String attributeName = instanceLines.get(i).split(" ")[1];
				String attributeType = instanceLines.get(i).split(" ")[2];
				this.attributeNames.add(attributeName);
				this.attributeTypes.add(attributeType);
			}
		}
		this.attributeNames.add("RUL");
		this.attributeTypes.add(Constants.ARFF_DATA_TYPE_NUMERIC);

		StringJoiner instanceJoiner = new StringJoiner("\n");
		for (int i = dataStartingIndex; i < instanceLines.size(); i++) {
			String[] sensorTimeseries = instanceLines.get(i).split(",");
			int instanceLength = this.getLengthOfInstance(sensorTimeseries);
			int rul = 0;
			int end = 1;
			if (instanceLength > MIN_REMAINING_TIMESTEPS_AFTER_FAILURE) {
				end = this.generateEndForInstance(instanceLength);
				rul = instanceLength - end;
			}

			StringJoiner sensorJoiner = new StringJoiner(",");
			for (int s = 0; s < sensorTimeseries.length; s++) {
				for (int t = 0; t < end; t++) {
					String[] sensorTimestepValuePairs = sensorTimeseries[s].replace("\"", "").trim().split(Constants.TIME_SERIES_DATA_POINT_SEPARATOR);
					String[] limitedSensorTimestepValueParis = Arrays.copyOfRange(sensorTimestepValuePairs, 0, end);
					sensorJoiner.add("\"" + String.join(Constants.TIME_SERIES_DATA_POINT_SEPARATOR, limitedSensorTimestepValueParis) + "\"");
				}
			}
			sensorJoiner.add(rul + "");
			instanceJoiner.add(sensorJoiner.toString());
			// }
		}

		return instanceJoiner.toString();
	}

	@Override
	public void transform(final String inputFileName, final String outputFileName) {
		LOGGER.info("Transforming input file: " + inputFileName);
		File outputFile = new File(outputFileName);
		outputFile.getParentFile().mkdirs();
		try (FileWriter fw = new FileWriter(outputFile)) {
			try (BufferedReader br = new BufferedReader(new FileReader(inputFileName))) {
				String line;
				boolean headerDone = false;
				while ((line = br.readLine()) != null) {
					if (!line.trim().isEmpty()) {
						// System.out.println(line);
						if (headerDone) {
							List<String> cuttedInstances = this.cutInstance(line);
							for (String copyCuttedInstance : cuttedInstances) {
								this.writeLine(fw, copyCuttedInstance);
							}
						} else if (line.startsWith(Constants.ARFF_HEADER_TAG_RELATION)) {
							this.writeLine(fw, line);
						} else if (line.startsWith(Constants.ARFF_HEADER_TAG_ATTRIBUTE)) {
							this.writeLine(fw, line);
						} else if (line.startsWith(Constants.ARFF_HEADER_TAG_DATA)) {
							this.writeLine(fw, Constants.ARFF_HEADER_TAG_ATTRIBUTE + " RUL " + Constants.ARFF_DATA_TYPE_NUMERIC);
							this.writeLine(fw, line);
							headerDone = true;
						}
					}
				}
			}
			fw.close();
		} catch (FileNotFoundException e) {
			LOGGER.log(Level.SEVERE, "Error writing to outputFile: " + outputFileName, e);
			e.printStackTrace();
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, "Error reading/writing to files.", e);
			e.printStackTrace();
		}
	}

	private void writeLine(final FileWriter fw, final String line) throws IOException {
		fw.write(line);
		fw.write("\n");
	}

	private List<String> cutInstance(final String instanceLine) {
		List<String> copiedInstances = new ArrayList<>(this.copyFactor);
		String[] sensorTimeseries = instanceLine.split(",");
		int instanceLength = this.getLengthOfInstance(sensorTimeseries);
		for (int copyId = 0; copyId < this.copyFactor; copyId++) {
			int rul = 0;
			int end = 1;
			if (instanceLength > MIN_REMAINING_TIMESTEPS_AFTER_FAILURE) {
				end = this.generateEndForInstance(instanceLength);
				rul = instanceLength - end;
				// System.out.println("rul: " + rul);
			}
			StringJoiner instanceJoiner = new StringJoiner(",");
			// if (this.timeSeriesStartingIndex > 0) {
			// if ("\"W300BNZ6\"".equals(sensorTimeseries[0])) {
			// String problem = "yes";
			// }
			// instanceJoiner.add(sensorTimeseries[0]);
			// instanceJoiner.add(sensorTimeseries[1]);
			// instanceJoiner.add(sensorTimeseries[2]);
			// }
			for (int s = this.timeSeriesStartingIndex; s < sensorTimeseries.length; s++) {
				if (sensorTimeseries[s].equals("?")) {
					instanceJoiner.add(String.join(Constants.TIME_SERIES_DATA_POINT_SEPARATOR, sensorTimeseries[s]));
				} else {
					String[] sensorTimestepValuePairs = sensorTimeseries[s].replace("\"", "").trim().split(Constants.TIME_SERIES_DATA_POINT_SEPARATOR);
					String[] limitedSensorTimestepValueParis = Arrays.copyOfRange(sensorTimestepValuePairs, 0, end);
					instanceJoiner.add("\"" + String.join(Constants.TIME_SERIES_DATA_POINT_SEPARATOR, limitedSensorTimestepValueParis) + "\"");
				}

			}
			instanceJoiner.add(rul + "");
			copiedInstances.add(instanceJoiner.toString());
		}
		return copiedInstances;
	}

	private int generateEndForInstance(final int timeseriesLength) {
		// int max = timeseriesLength - MIN_REMAINING_TIMESTEPS_AFTER_FAILURE;
		// int minimalLength = 0;
		// if (max - MAX_REMAINING_TIMESTEPS_AFTER_FAILURE >= 0) {
		// minimalLength = max - MAX_REMAINING_TIMESTEPS_AFTER_FAILURE;
		// max = max - (max - MAX_REMAINING_TIMESTEPS_AFTER_FAILURE);
		// }
		// if (max <= 0) {
		// max = MIN_REMAINING_TIMESTEPS_AFTER_FAILURE;
		// }
		// return minimalLength + this.random.nextInt(max) + 1;

		int maximalRUL = Math.min(MAX_REMAINING_TIMESTEPS_AFTER_FAILURE, timeseriesLength - 2);
		int minimalRUL = Math.min(MIN_REMAINING_TIMESTEPS_AFTER_FAILURE, timeseriesLength);

		int rul = this.random.nextInt((maximalRUL - minimalRUL) + 1) + minimalRUL;

		return timeseriesLength - rul;

	}

	private int getLengthOfInstance(final String[] sensors) {
		int largestTimeStepDiff = 0;
		for (int i = this.timeSeriesStartingIndex; i < sensors.length; i++) {
			String sensorValues = sensors[i].replace("\"", "").trim();
			String[] timestepValues = sensorValues.split(Constants.TIME_SERIES_DATA_POINT_SEPARATOR);
			String firstTimestepValue = timestepValues[0];
			String firstTimeStepStr = firstTimestepValue.split(Constants.TIME_STEP_VALUE_SEPARATOR)[0];
			String lastTimestepValue = timestepValues[timestepValues.length - 1];
			String lastTimeStepStr = lastTimestepValue.split(Constants.TIME_STEP_VALUE_SEPARATOR)[0];
			if (lastTimeStepStr.equals("?")) {
				continue;
			}
			int timeStepDiff = Integer.valueOf(lastTimeStepStr) - Integer.valueOf(firstTimeStepStr);
			timeStepDiff = timeStepDiff > timestepValues.length ? timestepValues.length : timeStepDiff;
			if (timeStepDiff > largestTimeStepDiff) {
				largestTimeStepDiff = timeStepDiff;
			}
		}
		return largestTimeStepDiff;
	}

	public String getFileExtension(final String filename) {
		if (filename.contains(".")) {
			return filename.substring(filename.lastIndexOf(".") + 1);
		}
		return null;
	}

	@Override
	public boolean canRead(final String inputFileName) {
		Path filePath = Paths.get(inputFileName);

		if (Files.exists(filePath, LinkOption.NOFOLLOW_LINKS) && Files.isRegularFile(filePath, LinkOption.NOFOLLOW_LINKS)) {
			if (this.getFileExtension(inputFileName) != null && this.getFileExtension(inputFileName).equals(Constants.ARFF_FILE_EXTENSION)) {
				return true;
			}
		}
		return false;
	}

}
