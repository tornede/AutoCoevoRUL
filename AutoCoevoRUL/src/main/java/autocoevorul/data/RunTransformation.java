package autocoevorul.data;

import java.io.File;

public class RunTransformation {

	private static final String PHM08_DATA_FOLDER_PATH = "/Users/hetzer/Downloads/Challenge_Data/";

	private static final String RAW_ARFF_DATA_FOLDER = "data_raw/";
	private static final String TRANSFORMED_ARFF_DATA_FOLDER = "data_transformed/";

	private static final String PHM_RAW_ARFF_DATA_FOLDER = RAW_ARFF_DATA_FOLDER + "PHM08DataChallenge/";
	private static final String PHM_TRANSFORMED_ARFF_DATA_FOLDER = TRANSFORMED_ARFF_DATA_FOLDER + "PHM08DataChallenge/";

	private static final String CMAPSS_DATA_FOLDER_PATH = "/Users/hetzer/Downloads/CMAPSSData/";
	private static final String CMAPSS_RAW_ARFF_DATA_FOLDER = RAW_ARFF_DATA_FOLDER + "CMAPSS/";
	private static final String CMAPSS_TRANSFORMED_ARFF_DATA_FOLDER = TRANSFORMED_ARFF_DATA_FOLDER + "CMAPSS/";

	public static void main(final String[] args) {
		transformCmapssToArffDataset();
		transformR2FtoRUL(CMAPSS_RAW_ARFF_DATA_FOLDER + "train/", CMAPSS_TRANSFORMED_ARFF_DATA_FOLDER);

		// transformPHM08Data();
		// transformR2FtoRUL(PHM_RAW_ARFF_DATA_FOLDER, PHM_TRANSFORMED_ARFF_DATA_FOLDER);

	}

	private static void transformPHM08Data() {
		Phm08DataChallengeToArffDatasetTransformer transformer = new Phm08DataChallengeToArffDatasetTransformer();

		transformer.transform(PHM08_DATA_FOLDER_PATH + "train.txt", PHM_RAW_ARFF_DATA_FOLDER + "train.arff");
		transformer.transform(PHM08_DATA_FOLDER_PATH + "test.txt", PHM_RAW_ARFF_DATA_FOLDER + "test.arff");
		transformer.transform(PHM08_DATA_FOLDER_PATH + "final_test.txt", PHM_RAW_ARFF_DATA_FOLDER + "final_test.arff");
	}

	private static void transformCmapssToArffDataset() {
		CmapssToArffDatasetTransformer tx = new CmapssToArffDatasetTransformer();
		if (!new File(CMAPSS_DATA_FOLDER_PATH).exists()) {
			System.out.println("cmapssDataFolderPath must be set");
		}

		tx.transform(CMAPSS_DATA_FOLDER_PATH + "test_FD001.txt", CMAPSS_RAW_ARFF_DATA_FOLDER + "test/FD001_test.arff");
		tx.transform(CMAPSS_DATA_FOLDER_PATH + "test_FD002.txt", CMAPSS_RAW_ARFF_DATA_FOLDER + "test/FD002_test.arff");
		tx.transform(CMAPSS_DATA_FOLDER_PATH + "test_FD003.txt", CMAPSS_RAW_ARFF_DATA_FOLDER + "test/FD003_test.arff");
		tx.transform(CMAPSS_DATA_FOLDER_PATH + "test_FD004.txt", CMAPSS_RAW_ARFF_DATA_FOLDER + "test/FD004_test.arff");
		tx.transform(CMAPSS_DATA_FOLDER_PATH + "train_FD001.txt", CMAPSS_RAW_ARFF_DATA_FOLDER + "train/FD001_train.arff");
		tx.transform(CMAPSS_DATA_FOLDER_PATH + "train_FD002.txt", CMAPSS_RAW_ARFF_DATA_FOLDER + "train/FD002_train.arff");
		tx.transform(CMAPSS_DATA_FOLDER_PATH + "train_FD003.txt", CMAPSS_RAW_ARFF_DATA_FOLDER + "train/FD003_train.arff");
		tx.transform(CMAPSS_DATA_FOLDER_PATH + "train_FD004.txt", CMAPSS_RAW_ARFF_DATA_FOLDER + "train/FD004_train.arff");
	}

	private static void transformR2FtoRUL(final String inputFolder, final String outputFolder) {
		File folder = new File(inputFolder);
		File[] files = folder.listFiles((f, name) -> name.endsWith("." + Constants.ARFF_FILE_EXTENSION));
		R2FtoLabeledTimeseriesDataTransformer tx = new R2FtoLabeledTimeseriesDataTransformer(0, 3);
		for (File file : files) {
			String fileName = file.getName();
			if (fileName.endsWith(".arff")) {
				tx.transform(file.getAbsolutePath(), outputFolder + fileName);
			}
		}
	}

}
