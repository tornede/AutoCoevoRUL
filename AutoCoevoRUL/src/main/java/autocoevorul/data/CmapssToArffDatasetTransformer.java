package autocoevorul.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Stream;

public class CmapssToArffDatasetTransformer implements DatasetTransformer {

	private static final Logger LOGGER = Logger.getLogger(CmapssToArffDatasetTransformer.class.getName());

	private static final int LINE_COUNT_TO_CHECK = 5; // used for canRead() function

	/**
	 * Reads given CMAPSS Data file, returns string in arff format
	 *
	 * @param inputFileName
	 * @return
	 */
	@Override
	public String transform(final String inputFileName) {
		Map<String, Map<Integer, String>> vals = null; // key=instanceNum-SensorNum
		List<String> ruls = null;
		String result = null;
		int sensorCount = 0;
		if (this.canRead(inputFileName)) {

			if (inputFileName.contains("test_")) {
				ruls = this.readRULs(inputFileName.replace("test_", "RUL_"));
			}

			vals = new LinkedHashMap<>();
			try (BufferedReader fileBufferReader = new BufferedReader(new FileReader(inputFileName))) {
				String line = null;
				while ((line = fileBufferReader.readLine()) != null) {
					String[] arr = line.split(" ");
					String instanceNum = arr[0];
					String timeStep = arr[1];
					sensorCount = arr.length - 2;
					for (int i = 2; i < arr.length; i++) {
						Map<Integer, String> timeStepToValue = vals.getOrDefault(instanceNum + "-" + (i - 1), new LinkedHashMap<>());
						timeStepToValue.put(Integer.parseInt(timeStep), arr[i]);
						vals.put(instanceNum + "-" + (i - 1), timeStepToValue);
					}

				}
			} catch (IOException e) {
				LOGGER.log(Level.SEVERE, "Error during file transformation: " + inputFileName, e);
			}

		}

		if (vals != null) {
			StringBuilder str = new StringBuilder();
			String currentInstance = null;
			for (Map.Entry<String, Map<Integer, String>> e : vals.entrySet()) {
				String[] instanceNumSensorNum = e.getKey().split("-");
				if (!instanceNumSensorNum[0].equals(currentInstance)) {
					if (currentInstance != null) {
						str.append("\n");
					}
					currentInstance = instanceNumSensorNum[0];
				}
				str.append("\"");
				for (Map.Entry<Integer, String> timeStepVal : e.getValue().entrySet()) {
					str.append(timeStepVal.getKey()).append(Constants.TIME_STEP_VALUE_SEPARATOR);
					str.append(timeStepVal.getValue()).append(Constants.TIME_SERIES_DATA_POINT_SEPARATOR);
				}
				str.replace(str.length() - 1, str.length(), "");
				str.append("\",");

				// append RUL value if exists
				if (instanceNumSensorNum[1].equals(String.valueOf(sensorCount))) {
					if (ruls != null) {
						str.append(" ").append(ruls.get(Integer.parseInt(instanceNumSensorNum[0]) - 1));
					} else {
						str.replace(str.length() - 1, str.length(), "");
					}
				}
			}
			result = str.toString();
		}

		return result;
	}

	/**
	 * Reads given CMAPSS Data file, writes transformed arff format to given output file
	 *
	 * @param inputFileName
	 * @param outputFileName
	 */
	@Override
	public void transform(final String inputFileName, final String outputFileName) {
		StringBuilder str = new StringBuilder();
		String simpleName = this.extractSimpleFileName(inputFileName);
		str.append(Constants.ARFF_HEADER_TAG_RELATION).append(" CMAPSS_" + simpleName + "\n\n");
		for (int i = 0; i < 24; i++) {
			str.append(Constants.ARFF_HEADER_TAG_ATTRIBUTE).append(" sensor").append(i).append(" ").append(Constants.ARFF_DATA_TYPE_TIMESERIES).append(" \n");
		}
		if (inputFileName.contains("test_")) {
			str.append(Constants.ARFF_HEADER_TAG_ATTRIBUTE).append(" RUL ").append(Constants.ARFF_DATA_TYPE_NUMERIC).append(" \n");
		}
		str.append("\n");
		str.append(Constants.ARFF_HEADER_TAG_DATA);
		str.append("\n");
		String output = this.transform(inputFileName);
		str.append(output);

		File file = new File(outputFileName);
		try {
			file.getParentFile().mkdirs();
			file.createNewFile();
		} catch (IOException e1) {
			e1.printStackTrace();
		}

		try (PrintWriter out = new PrintWriter(file)) {
			out.print(str);
		} catch (FileNotFoundException e) {
			LOGGER.log(Level.SEVERE, "Error writing to outputFile: " + outputFileName, e);
		}

	}

	/**
	 * Check whether the given CMAPSS Data file exists, and contains correct amount of sensor data
	 *
	 *
	 * @param inputFileName
	 * @return
	 */
	@Override
	public boolean canRead(final String inputFileName) {
		Path filePath = Paths.get(inputFileName);

		if (Files.exists(filePath, LinkOption.NOFOLLOW_LINKS) && Files.isRegularFile(filePath, LinkOption.NOFOLLOW_LINKS)) {
			try (Stream<String> lines = Files.lines(filePath)) {
				return lines.limit(LINE_COUNT_TO_CHECK).allMatch(e -> this.isCorrectFormat(e));
			} catch (IOException e) {
				LOGGER.log(Level.SEVERE, "can't read input file: " + inputFileName, e);
				return false;
			}
		}
		return false;
	}

	private List<String> readRULs(final String rulFileName) {
		List<String> ruls = new ArrayList<>();
		try (BufferedReader fileBufferReader = new BufferedReader(new FileReader(rulFileName))) {
			String line = null;
			while ((line = fileBufferReader.readLine()) != null) {
				String[] arr = line.split(" ");
				for (int i = 0; i < arr.length; i++) {
					ruls.add(arr[i]);
				}
			}
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, "Error reading RUL file: " + rulFileName, e);
		}
		return ruls;
	}

	private boolean isCorrectFormat(final String line) {
		// CMAPSS Dataset contains 26 columns
		String[] columns = line.split(" ");
		return columns.length == 26;
	}

	private String extractSimpleFileName(final String inputFileName) {
		int beginIndex = inputFileName.lastIndexOf(File.separator) + 1;
		int endIndex;
		if (inputFileName.substring(beginIndex).contains(".")) {
			endIndex = inputFileName.lastIndexOf(".");
		} else {
			endIndex = inputFileName.length();
		}
		String simpleName = inputFileName.substring(beginIndex, endIndex);
		return simpleName;
	}

}
