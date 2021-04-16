package autocoevorul;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import ai.libs.jaicore.components.api.IComponentRepository;
import ai.libs.jaicore.components.serialization.ComponentSerialization;

public class ScikitLearnSearchSpaceDeserializationTest {

	private static final String BASE_PATH = "src/main/resources/searchspace/";

	@ParameterizedTest
	@MethodSource("provideRepositoriesToTest")
	public void testDeserializationOfRepository(final String path, final int numExpectedComponents) throws IOException {
		Map<String, String> templateVars = new HashMap<>();
		templateVars.put("instance_lengt", "10");
		File file = new File(path);
		IComponentRepository repo = new ComponentSerialization().deserializeRepository(file, templateVars);
		assertEquals(numExpectedComponents, repo.size(),
				String.format("Number of components deserialized from path %s is %s instead of the expected number %s ", path, repo.size(), numExpectedComponents));
	}

	public static Stream<Arguments> provideRepositoriesToTest() {
		return Stream.of(
				/* Timeseries Feature Extraction */
				Arguments.of(BASE_PATH + "timeseries/pyts/transformation/BagOfPatterns.json", 1), //
				Arguments.of(BASE_PATH + "timeseries/pyts/transformation/Boss.json", 1), //
				Arguments.of(BASE_PATH + "timeseries/pyts/transformation/Rocket.json", 1), //
				Arguments.of(BASE_PATH + "timeseries/pyts/transformation/ShapeletTransform.json", 1), //
				Arguments.of(BASE_PATH + "timeseries/pyts/transformation/Weasel.json", 1), //
				Arguments.of(BASE_PATH + "timeseries/pyts/transformation/UltraFastShapelets.json", 1), //
				Arguments.of(BASE_PATH + "timeseries/pyts/pyts.json", 10), //

				Arguments.of(BASE_PATH + "timeseries/tsfresh/featureset/FCParametersDictionary.json", 1), //
				Arguments.of(BASE_PATH + "timeseries/tsfresh/featureset/LowComputationTimeFCParameters.json", 1), //
				Arguments.of(BASE_PATH + "timeseries/tsfresh/featureset/MidComputationTimeFCParameters.json", 1), //
				Arguments.of(BASE_PATH + "timeseries/tsfresh/tsfresh.json", 4), //

				Arguments.of(BASE_PATH + "timeseries/index.json", 14), //

				/* Masking Strategies */
				Arguments.of(BASE_PATH + "timeseries/pyts/masking/index.json", 3), //

				/* Regressors */
				Arguments.of(BASE_PATH + "regression/mlpregressor.json", 1), //
				Arguments.of(BASE_PATH + "regression/index.json", 8), //

				/* RUL */
				Arguments.of(BASE_PATH + "rul.json", 23) //
		);
	}

}
