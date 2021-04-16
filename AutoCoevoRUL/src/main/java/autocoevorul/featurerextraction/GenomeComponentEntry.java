package autocoevorul.featurerextraction;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.moeaframework.core.Variable;
import org.moeaframework.core.variable.BinaryIntegerVariable;
import org.moeaframework.core.variable.RealVariable;
import org.slf4j.Logger;

import ai.libs.jaicore.components.api.IParameter;
import ai.libs.jaicore.components.model.CategoricalParameterDomain;
import ai.libs.jaicore.components.model.NumericParameterDomain;

public class GenomeComponentEntry {

	private static Logger LOGGER = org.slf4j.LoggerFactory.getLogger(GenomeComponentEntry.class);

	private final int activationIndex;
	private final String name;
	private final Map<Integer, IParameter> indexToParametersMap;
	private final Map<String, String> singleValueParametersMap;

	public GenomeComponentEntry(final int index, final String name) {
		super();
		this.activationIndex = index;
		this.name = name;
		this.indexToParametersMap = new HashMap<>();
		this.singleValueParametersMap = new HashMap<>();
	}

	public GenomeComponentEntry(final String name) {
		this(-1, name);
	}

	public boolean hasActivationBit() {
		return this.activationIndex >= 0;
	}

	public int getActivationIndex() {
		return this.activationIndex;
	}

	public Set<Integer> getParameterIndices() {
		return this.indexToParametersMap.keySet();
	}

	public String getName() {
		return this.name;
	}

	public int getNumberOfParameters() {
		return this.indexToParametersMap.size();
	}

	public IParameter getParameter(final int index) {
		return this.indexToParametersMap.get(index);
	}

	public Map<String, String> getSingleValueCategoricalParametersMap() {
		return this.singleValueParametersMap;
	}

	public boolean addParameter(final int index, final IParameter parameter) {
		if (parameter.isCategorical()) {
			CategoricalParameterDomain domain = (CategoricalParameterDomain) parameter.getDefaultDomain();
			if (domain.getValues().length > 1) {
				this.indexToParametersMap.put(index, parameter);
				return true;
			} else {
				this.singleValueParametersMap.put(parameter.getName(), domain.getValues()[0]);
			}
		} else {
			NumericParameterDomain numericParameter = (NumericParameterDomain) parameter.getDefaultDomain();
			if (numericParameter.getMin() < numericParameter.getMax()) {
				this.indexToParametersMap.put(index, parameter);
				return true;
			} else {
				this.singleValueParametersMap.put(parameter.getName(), "" + numericParameter.getMin());
			}
		}
		return false;
	}

	public Variable getParameterVariable(final int index) {
		IParameter parameter = this.indexToParametersMap.get(index);
		if (parameter.isCategorical()) {
			CategoricalParameterDomain domain = (CategoricalParameterDomain) parameter.getDefaultDomain();
			LOGGER.debug("({}) {}#{}={}", index, this.name, parameter.getName(), Arrays.toString(domain.getValues()));
			return new BinaryIntegerVariable(0, domain.getValues().length - 1);
		} else {
			NumericParameterDomain numericParameter = (NumericParameterDomain) parameter.getDefaultDomain();
			if (numericParameter.isInteger()) {
				LOGGER.debug("({}) {}#{}={}-{}", index, this.name, parameter.getName(), (int) numericParameter.getMin(), (int) numericParameter.getMax());
				return new BinaryIntegerVariable((int) numericParameter.getMin(), (int) numericParameter.getMax());
			}
			LOGGER.debug("({}) {}#{}={}-{}", index, this.name, parameter.getName(), numericParameter.getMin(), numericParameter.getMax());
			return new RealVariable(numericParameter.getMin(), numericParameter.getMax());
		}
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + this.activationIndex;
		result = prime * result + ((this.indexToParametersMap == null) ? 0 : this.indexToParametersMap.hashCode());
		result = prime * result + ((this.name == null) ? 0 : this.name.hashCode());
		return result;
	}

	@Override
	public boolean equals(final Object obj) {
		if (this == obj) {
			return true;
		}
		if (obj == null) {
			return false;
		}
		if (this.getClass() != obj.getClass()) {
			return false;
		}
		GenomeComponentEntry other = (GenomeComponentEntry) obj;
		if (this.activationIndex != other.activationIndex) {
			return false;
		}
		if (this.indexToParametersMap == null) {
			if (other.indexToParametersMap != null) {
				return false;
			}
		} else if (!this.indexToParametersMap.equals(other.indexToParametersMap)) {
			return false;
		}
		if (this.name == null) {
			if (other.name != null) {
				return false;
			}
		} else if (!this.name.equals(other.name)) {
			return false;
		}
		return true;
	}

	@Override
	public String toString() {
		return "ComponentEntry [index=" + this.activationIndex + ", name=" + this.name + ", integerToParametersMap=" + this.indexToParametersMap + "]";
	}

}
