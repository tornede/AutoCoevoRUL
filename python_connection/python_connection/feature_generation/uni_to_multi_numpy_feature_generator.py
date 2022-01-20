from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import copy
import random


class MaskingStrategy:
    pass


class ZeroMaskingStrategy(MaskingStrategy):

    def extend_instance(self, instance: list, maximum_length: int):
        while len(instance) < maximum_length:
            instance.insert(0, 0)


class FirstValueMaskingStrategy(MaskingStrategy):

    def extend_instance(self, instance: list, maximum_length: int):
        first_value = instance[0]
        while len(instance) < maximum_length:
            instance.insert(0, first_value)


class RandomValueMaskingStrategy(MaskingStrategy):

    def extend_instance(self, instance: list, maximum_length: int):
        instance_as_numpy = np.asarray(instance)
        minimum_value = instance_as_numpy.min()
        maximum_value = instance_as_numpy.max()

        while len(instance) < maximum_length:
            random_value = minimum_value if minimum_value == maximum_value else \
                np.random.uniform(minimum_value, maximum_value, 1)[0]
            instance.insert(0, random_value)


"""
This class is a wrapper for feature generators which work with univariate timeseries and on a numpy array where each
row corresponds to an instance and each column to a timestep. Furthermore the generator is assumed to work with timeseries
of equal length and with a value for each timestep. 

To abide to the constraints described above, this wrapper enlarges all timeseries such that they are as long as the longest
one in the dataset. The additionally created timesteps are filled according to the masking strategy. 

Moreover, to make the univariate methods generate features for multivariate timeseries, the generators are called once
for each sensor and the corresponding feature matrices are concatenated horizontally.
"""


class UniToMultivariateNumpyBasedFeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self, univariate_ts_feature_generator, masking_strategy: MaskingStrategy = ZeroMaskingStrategy()):
        self.pyts_transformer = univariate_ts_feature_generator
        self.masking_strategy = masking_strategy
        self.attribute_id_to_transformer_map = dict()
        self.maximum_instance_length = None

    def fit(self, X, y):
        attribute_id_to_dataset_map = self.transform_arff_parsed(X, training=True)
        for attribute_id, dataset in attribute_id_to_dataset_map.items():
            transformer_for_attribute_id = copy.deepcopy(self.pyts_transformer)
            transformer_for_attribute_id.fit(dataset[0], dataset[1])
            self.attribute_id_to_transformer_map[attribute_id] = transformer_for_attribute_id

        return self

    def transform(self, X):
        attribute_id_to_dataset_map = self.transform_arff_parsed(X, training=False)
        complete_dataset = None
        attribute_ids = list(attribute_id_to_dataset_map.keys())
        attribute_ids.sort()
        for attribute_id in attribute_ids:
            dataset = attribute_id_to_dataset_map[attribute_id]
            transformer_for_attribute_id = self.attribute_id_to_transformer_map[attribute_id]

            transformed_dataset_for_attribute_id = transformer_for_attribute_id.transform(dataset[0])
            if complete_dataset is None:
                complete_dataset = transformed_dataset_for_attribute_id
            else:
                complete_dataset = np.hstack((complete_dataset, transformed_dataset_for_attribute_id))

        return complete_dataset

    def transform_arff_parsed(self, arff_parsed, training: bool):
        attributes = arff_parsed['attributes']

        sensor_to_dataset_map = dict()

        sensor_attribute_ids = [attribute_id for attribute_id in range(len(attributes)) if
                                'sensor' in attributes[attribute_id][0].lower()]
        target_attribute_id = \
            [attribute_id for attribute_id in range(len(attributes)) if 'rul' == attributes[attribute_id][0].lower()][0]
        if self.maximum_instance_length is None:
            self.maximum_instance_length = np.zeros(len(sensor_attribute_ids))

        for attribute_index, attribute_id in enumerate(sensor_attribute_ids):
            dataset_for_sensor = list()
            targets_for_sensor = list()
            for instance_id, instance in enumerate(arff_parsed['data']):
                raw_attribute_value = instance[attribute_id]

                split_attribute_value = raw_attribute_value.split()
                instance_length = len(split_attribute_value)
                if instance_length > self.maximum_instance_length[attribute_index]:
                    self.maximum_instance_length[attribute_index] = instance_length

                instance_for_attribute = list()
                for time_step_value_pair in split_attribute_value:
                    if '#' not in time_step_value_pair:
                        raise Exception("INVALID FORMAT!")
                    else:
                        # TODO ASSUMPTION: EACH INSTANCE HAS A VALUE FOR EACH TIMESTEP OF THE CORRESPONDING SENSOR
                        split_pair = time_step_value_pair.split('#')
                        value = float(split_pair[1])  # TODO robustness
                        instance_for_attribute.append(value)
                dataset_for_sensor.append(instance_for_attribute)

                target = instance[target_attribute_id]
                targets_for_sensor.append(target)

            # make sure that all instances have same length and add dummy entry at end for shorter ones
            for instance in dataset_for_sensor:
                self.masking_strategy.extend_instance(instance, self.maximum_instance_length[attribute_index])

            sensor_to_dataset_map[attribute_id] = np.array(dataset_for_sensor), np.array(targets_for_sensor)

        return sensor_to_dataset_map
