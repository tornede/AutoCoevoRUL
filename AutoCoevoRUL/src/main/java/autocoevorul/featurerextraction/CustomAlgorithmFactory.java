package autocoevorul.featurerextraction;

import java.util.Properties;
import java.util.ServiceConfigurationError;

import org.moeaframework.core.Algorithm;
import org.moeaframework.core.Initialization;
import org.moeaframework.core.Problem;
import org.moeaframework.core.spi.AlgorithmFactory;
import org.moeaframework.core.spi.AlgorithmProvider;

public class CustomAlgorithmFactory extends AlgorithmFactory {

	private Initialization initialization;

	public CustomAlgorithmFactory(final Initialization initialization) {
		this.initialization = initialization;
	}

	/**
	 * Attempts to instantiate the given algorithm using the given provider.
	 *
	 * @param provider the algorithm provider
	 * @param name the name identifying the algorithm
	 * @param properties the implementation-specific properties
	 * @param problem the problem to be solved
	 * @return an instance of the algorithm with the registered name; or
	 *         {@code null} if the provider does not implement the algorithm
	 */
	@Override
	protected Algorithm instantiateAlgorithm(final AlgorithmProvider provider, final String name, final Properties properties, final Problem problem) {
		try {
			return provider.getAlgorithm(name, properties, problem, this.initialization);
		} catch (ServiceConfigurationError e) {
			System.err.println(e.getMessage());
		}

		return null;
	}
}
