package autocoevorul;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import ai.libs.jaicore.components.api.IComponent;
import ai.libs.jaicore.components.api.IComponentRepository;
import ai.libs.jaicore.components.serialization.ComponentSerialization;

public class ScikitLearnSearchSpaceDeserializationTest {

	public ScikitLearnSearchSpaceDeserializationTest() throws NoSuchMethodException, SecurityException, InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		// super(BinaryAttributeSelectionIncludedGenomeHandler.class);
	}

	private static final String BASE_PATH = "src/main/resources/searchspace/";

	@ParameterizedTest
	@MethodSource("provideRepositoriesToTest")
	public void testDeserializationOfRepository(final String path, final int numExpectedComponents) throws IOException {
		final Map<String, String> templateVars = new HashMap<>();
		templateVars.put("instance_lengt", "10");
		final File file = new File(path);
		final IComponentRepository repo = new ComponentSerialization().deserializeRepository(file, templateVars);
		for (final IComponent component : repo) {
			System.out.println(component);
		}
		assertEquals(numExpectedComponents, repo.size(), String.format("Number of components deserialized from path %s is %s instead of the expected number %s ", path, repo.size(), numExpectedComponents));
	}

	public static Stream<Arguments> provideRepositoriesToTest() {
		return Stream.of(
				/* attribute filter */
				Arguments.of(BASE_PATH + "timeseries/attributes/attribute_filter.json", 1), //
				Arguments.of(BASE_PATH + "timeseries/attributes/attribute_types.json", 3), //
				Arguments.of(BASE_PATH + "timeseries/attributes/index.json", 4), //

				/* tsfresh */
				Arguments.of(BASE_PATH + "timeseries/tsfresh/tsfresh_features.json", 72), //
				Arguments.of(BASE_PATH + "timeseries/tsfresh/tsfresh.json", 1), //
				Arguments.of(BASE_PATH + "timeseries/tsfresh/index.json", 73), //

				/* all timeseries feature elements */
				Arguments.of(BASE_PATH + "timeseries/index.json", 77), //

				/* Regressors */
				Arguments.of(BASE_PATH + "regression/mlpregressor.json", 1), //
				Arguments.of(BASE_PATH + "regression/index.json", 8), //

				/* RUL */
				Arguments.of(BASE_PATH + "rul.json", 86) //
		);
	}

}
