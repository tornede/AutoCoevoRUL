import json 
import os
from ml4pdm.transformation.fixed_size import TsfreshFeature, TsfreshWrapper
import errno


PATH_SEARCHSPACE = os.path.join('searchspace_extraction')
PATH_TIMESERIES = os.path.join(PATH_SEARCHSPACE,'timeseries')

REPOSITORY_ML4PDM = 'ml4pdm'
REPOSITORY_TRANSFORMATION = REPOSITORY_ML4PDM + '.transformation'

INTERFACE_TIMESERIES_TO_TABULAR_FEATURE_ENGINEERING = 'TimeseriesToTabularFeatureEngineering'


def mkdir(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as exc: # Python >2.5
        if not (exc.errno == errno.EEXIST and os.path.isdir(os.path.dirname(path))):
            raise
        
        
def get_module_and_class_name(o):
    klass = o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__ # avoid outputs like 'builtins.str'
    return module, klass.__qualname__


def create_required_interface(identifier, name, min, max, unique=False):
    return {"id": identifier, "name": name, "min": min, "max": max, "unique": 'true' if unique else 'false'}


def create_component(name, provided_interface=[], required_interface=[], parameters=[]):
    return {"name": name, "providedInterface": provided_interface, "requiredInterface": required_interface, "parameters": parameters}


def write_repository(file, name, includes= [], components=[]): 
    if not file.endswith('.json'):
        file += '.json'
    mkdir(file)
    component = {'repository': name, 'include': includes, 'components': components}
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(component, f, ensure_ascii=False, indent=4)


def extract_tsfresh():
    tsfresh = 'tsfresh'
    interface_tsfresh_feature_name = "TsfreshFeature"
    interface_tsfresh_feature_identifier = "tsfresh_features"
    path_tsfresh = os.path.join(PATH_TIMESERIES, tsfresh)
    
    tsfresh_wrapper_module, tsfresh_wrapper_class = get_module_and_class_name(TsfreshWrapper())
    write_repository(
        file = os.path.join(path_tsfresh, tsfresh),
        name = REPOSITORY_TRANSFORMATION,
        components = [create_component(
            name = f"{tsfresh_wrapper_module}.{tsfresh_wrapper_class}",
            provided_interface=[INTERFACE_TIMESERIES_TO_TABULAR_FEATURE_ENGINEERING],
            required_interface=[create_required_interface(
                identifier = interface_tsfresh_feature_identifier, 
                name = interface_tsfresh_feature_name, 
                min =  1, 
                max = 2, 
                unique = True
            )]
        )
    ])
    
    tsfresh_feature_components=[]
    for tsfresh_feature in TsfreshFeature:
        tsfresh_feature_module, tsfresh_feature_class = get_module_and_class_name(tsfresh_feature)
        tsfresh_feature_components.append(create_component(
            name = f"{tsfresh_feature_module}.{tsfresh_feature}",
            provided_interface = [interface_tsfresh_feature_name]))
    write_repository(
        file = os.path.join(path_tsfresh, 'tsfresh_features'), 
        name = REPOSITORY_TRANSFORMATION, 
        components = tsfresh_feature_components
    )


if __name__ == "__main__":
    extract_tsfresh()
