package autocoevorul.experiment.results;

import java.io.File;
import java.sql.SQLException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.aeonbits.owner.ConfigFactory;
import org.api4.java.datastructure.kvstore.IKVStore;

import ai.libs.jaicore.basic.StatisticsUtil;
import ai.libs.jaicore.basic.ValueUtil;
import ai.libs.jaicore.basic.kvstore.KVStore;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection.EGroupMethod;
import ai.libs.jaicore.basic.kvstore.KVStoreSequentialComparator;
import ai.libs.jaicore.basic.kvstore.KVStoreStatisticsUtil;
import ai.libs.jaicore.basic.kvstore.KVStoreUtil;
import ai.libs.jaicore.db.IDatabaseAdapter;
import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.db.sql.DatabaseAdapterFactory;

public class ResultsTable {

	private static final File configFile = new File("conf/experiments/coevolution.properties");
	private static final IDatabaseConfig dbconfig = (IDatabaseConfig) ConfigFactory.create(IDatabaseConfig.class).loadPropertiesFromFile(configFile);

	private static final String COEVOLUTON_TABLE = "8fixedInit";
	private static final String MLPLAN_TABLE = "3";
	private static final String RANDOM_SEARCH_TABLE = "3";

	private static final String INTERNAL_PERFORMANCE_MEASURE_ASYM = "ASYMMETRIC_LOSS";
	private static final String INTERNAL_PERFORMANCE_MEASURE_MAPE = "MEAN_ABSOLUTE_PERCENTAGE_ERROR";
	private static final String INTERNAL_FEATURE_OBJECTIVE_MEASURE = "MINIMUM"; // MINIMUM AVERAGE MEDIAN
	private static final String FINAL_PERFORMANCE_MEASURE = "performance_asymmetric_loss"; // performance_asymmetric_loss_mean performance_mean_absolute_percentage_error

	private static final String SEEDS = "(1,2,3,4,5)";

	public static void main(final String[] args) throws SQLException {
		KVStoreCollection x = get(INTERNAL_PERFORMANCE_MEASURE_ASYM);
		x.addAll(get(INTERNAL_PERFORMANCE_MEASURE_MAPE));

		String latexTable = KVStoreUtil.kvStoreCollectionToLaTeXTable(x, "datasetName", "approach", "entry");
		latexTable = latexTable.replaceAll("_train.arff", "");
		latexTable = latexTable.replaceAll("/train.arff", "");
		latexTable = latexTable.replaceAll("Phm", "PHM");
		latexTable = latexTable.replaceAll("DataChallenge", "");
		latexTable = latexTable.replaceAll("mlplan", "ML-Plan-RUL");
		latexTable = latexTable.replaceAll("rs", "RS");
		latexTable = latexTable.replaceAll("AVERAGE", "AutoCoevoRUL");
		latexTable = latexTable.replaceAll("MINIMUM", "AutoCoevoRUL");
		latexTable = latexTable.replaceAll("MEDIAN", "AutoCoevoRUL");
		latexTable = latexTable.replaceAll("ASYMMETRIC_LOSS", "\\$\\\\mathcal{L}_{\\\\mathit{assym}}\\$");
		latexTable = latexTable.replaceAll("MEAN_ABSOLUTE_PERCENTAGE_ERROR", "\\$\\\\mathcal{L}_{\\\\mathit{MAPE}}\\$");

		System.out.println(latexTable);
	}

	public static KVStoreCollection get(final String internalPerformanceMeasure) throws SQLException {
		try (IDatabaseAdapter db = DatabaseAdapterFactory.get(dbconfig)) {
			KVStoreCollection col = KVStoreUtil
					.readFromMySQLQuery(db,
							"SELECT datasetName, internal_performance_measure, featureObjectiveMeasure, seed, internal_performance, performance_asymmetric_loss, performance_mean_absolute_percentage_error FROM coevolutionV"
									+ COEVOLUTON_TABLE + " WHERE seed in " + SEEDS + " AND internal_performance_measure = '" + internalPerformanceMeasure + "' AND featureObjectiveMeasure = '" + INTERNAL_FEATURE_OBJECTIVE_MEASURE + "'",
							new HashMap<>());
			col.merge(new String[] { "internal_performance_measure", "featureObjectiveMeasure" }, "/", "approach");
			col.stream().forEach(x -> {
				for (String m : Arrays.asList("performance_mean_absolute_percentage_error", "performance_asymmetric_loss", "internal_performance")) {
					if (!isNumeric(x.get(m))) {
						x.put(m, 1.0);
					}
				}
			});

			Map<String, String> fields = new HashMap<>();
			fields.put("method", "rs");
			KVStoreCollection colRS = KVStoreUtil.readFromMySQLQuery(db, "SELECT * FROM randomSearchV" + RANDOM_SEARCH_TABLE + " WHERE seed in " + SEEDS + " AND internal_performance_measure = '" + internalPerformanceMeasure + "'", fields);
			colRS.merge(new String[] { "internal_performance_measure", "method" }, "/", "approach");
			colRS.stream().forEach(x -> {
				for (String m : Arrays.asList("performance_mean_absolute_percentage_error", "performance_asymmetric_loss", "internal_performance")) {
					if (!isNumeric(x.get(m))) {
						x.put(m, 1.0);
					}
				}
			});

			col.addAll(colRS);

			fields.put("method", "mlplan");
			KVStoreCollection colMLPlan = KVStoreUtil.readFromMySQLQuery(db, "SELECT * FROM mlplanV" + MLPLAN_TABLE + " WHERE seed in " + SEEDS + " AND internal_performance_measure = '" + internalPerformanceMeasure + "'", fields);
			colMLPlan.merge(new String[] { "internal_performance_measure", "method" }, "/", "approach");
			colMLPlan.stream().forEach(x -> {
				for (String m : Arrays.asList("performance_mean_absolute_percentage_error", "performance_asymmetric_loss", "internal_performance")) {
					if (!isNumeric(x.get(m))) {
						x.put(m, 1.0);
					}
				}
			});

			col.addAll(colMLPlan);

			Map<String, EGroupMethod> grouping = new HashMap<>();
			grouping.put("performance_mean_absolute_percentage_error", EGroupMethod.AVG);
			grouping.put("internal_performance", EGroupMethod.AVG);
			grouping.put("performance_asymmetric_loss", EGroupMethod.AVG);
			KVStoreCollection grouped = col.group("datasetName", "approach");

			for (String m : Arrays.asList("performance_mean_absolute_percentage_error", "performance_asymmetric_loss", "internal_performance")) {
				grouped.stream().forEach(x -> x.put(m + "_mean", ValueUtil.valueToString(StatisticsUtil.mean(x.getAsDoubleList(m)), 4)));
				grouped.stream().forEach(x -> x.put(m + "_std", ValueUtil.valueToString(StatisticsUtil.standardDeviation(x.getAsDoubleList(m)), 2)));
			}
			grouped.stream().forEach(x -> x.put("entry", "#" + FINAL_PERFORMANCE_MEASURE + "_mean# $\\pm$ #" + FINAL_PERFORMANCE_MEASURE + "_std# (#rank#)"));

			grouped.sort(new KVStoreSequentialComparator("datasetName", "approach"));

			KVStoreStatisticsUtil.rank(grouped, "datasetName", "approach", FINAL_PERFORMANCE_MEASURE + "_mean", "rank");
			KVStoreStatisticsUtil.averageRank(grouped, "approach", "rank").entrySet().stream().forEach(x -> {
				IKVStore s = new KVStore();
				s.put("approach", x.getKey());
				s.put("datasetName", "\\midrule \navg. Rank");
				s.put("entry", x.getValue().getMean());
				grouped.add(s);
			});
			return grouped;
		}

	}

	public static boolean isNumeric(final Object strNum) {
		if (strNum == null) {
			return false;
		}
		try {
			Double.parseDouble(strNum.toString());
		} catch (NumberFormatException nfe) {
			return false;
		}
		return true;
	}

}
