package autocoevorul.util;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import ai.libs.jaicore.basic.ResourceFile;
import ai.libs.jaicore.components.api.IComponent;
import ai.libs.jaicore.components.model.ComponentUtil;
import ai.libs.jaicore.components.serialization.ComponentSerialization;

public class ComponentCollectionUtil {

	public static List<IComponent> getAllComponents(final String searchSpace, final Map<String, String> templateVariables) throws IOException {
		return new ComponentSerialization().deserializeRepository(new ResourceFile(searchSpace), templateVariables).stream().map(c -> c).sorted((c1, c2) -> c1.getName().compareTo(c2.getName())).collect(Collectors.toList());
	}

	public static List<IComponent> getComponentsSatisfyingInterface(final String searchSpace, final String requiredInterface, final Map<String, String> templateVariables) throws IOException {
		ComponentSerialization componentSerialization = new ComponentSerialization();
		return ComponentUtil.getComponentsProvidingInterface(componentSerialization.deserializeRepository(new ResourceFile(searchSpace), templateVariables), requiredInterface).stream().map(c -> c)
				.sorted((c1, c2) -> c1.getName().compareTo(c2.getName())).collect(Collectors.toList());
	}
}
