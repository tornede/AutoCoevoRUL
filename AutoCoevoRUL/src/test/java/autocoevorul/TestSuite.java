package autocoevorul;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

@RunWith(Suite.class)

@Suite.SuiteClasses({ CoevolutionTest.class, MoeaExampleProblemTest.class, MoeaSolutionEncodingTest.class, TSFreshFeatureGenerationTest.class })
public class TestSuite {

}
