package autocoevorul.data;

public interface DatasetTransformer {

	/**
	 * Reads given input file, returns transformed format as string
	 *
	 * @param inputFileName
	 * @return
	 */
	public String transform(String inputFileName);

	/**
	 * Reads given input file, writes transformed format to given output file
	 *
	 * @param inputFileName
	 * @param outputFileName
	 */
	public void transform(String inputFileName, String outputFileName);

	/**
	 * Check whether the input file is in correct format
	 *
	 * @param inputFileName
	 * @return
	 */
	public boolean canRead(String inputFileName);

}
