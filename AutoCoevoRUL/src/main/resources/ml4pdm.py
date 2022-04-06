import warnings
from sklearn.pipeline import make_union
from sklearn.pipeline import make_pipeline
import numpy as np
import sys
import json
import argparse
import resource
import os

from joblib import dump, load

from ml4pdm.data import DatasetParser

def warn(*args, **kwargs):
    pass

warnings.warn = warn

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


{{imports}}


class ProblemType:
    TIME_SERIES_REGRESSION = 'ts-reg'
    TIME_SERIES_FEATURE_ENGINEERING = 'ts-fe'

    @staticmethod
    def get(identifier):
        if identifier == ProblemType.TIME_SERIES_REGRESSION:
            return ProblemType.TIME_SERIES_REGRESSION
        elif identifier == ProblemType.TIME_SERIES_FEATURE_ENGINEERING:
            return ProblemType.TIME_SERIES_FEATURE_ENGINEERING
        else:
            raise RuntimeError("Unsupported problem type: " + identifier)


class ModeType:
    FIT = 'fit'
    PREDICT = 'predict'
    FIT_AND_PREDICT = 'fitAndPredict'

    @staticmethod
    def get(identifier):
        if identifier == ModeType.FIT:
            return ModeType.FIT
        elif identifier == ModeType.PREDICT:
            return ModeType.PREDICT
        elif identifier == ModeType.FIT_AND_PREDICT:
            return ModeType.FIT_AND_PREDICT
        else:
            raise RuntimeError("Unsupported mode: " + identifier)


class ArgsHandler:

    @staticmethod
    def setup():
        """
        Parses the arguments that are given to the script and overwrites sys.argv with this parsed representation that is
        accessable as a list.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--problem', choices=[ProblemType.TIME_SERIES_REGRESSION, ProblemType.TIME_SERIES_FEATURE_ENGINEERING], required=True, help="")
        parser.add_argument('--mode', choices=['fit', 'predict', 'fitAndPredict'], required=True, help="Selecting whether a train or a test is run.")
        parser.add_argument('--fit', help="Path or data to use for training.")
        parser.add_argument('--fitOutput', help="In train mode set the file where the model shall be dumped; in test mode set the file where the prediction results shall be serialized to.")
        parser.add_argument('--predict', help="Path or data to use for testing when running with predict mode or fitAndPredict mode.")
        parser.add_argument('--predictOutput', help="In train mode set the file where the model shall be dumped; in test mode set the file where the prediction results shall be serialized to.")
        parser.add_argument('--model', help="Path to the trained model (in .pcl format) that shall be used for testing.")
        parser.add_argument('--targets', nargs='*', type=int, help="Declare which of the columns of the ARFF to use as targets. Default is only the last column.")
        parser.add_argument('--seed', required=True, help="Sets the seed.")
        sys.argv = vars(parser.parse_args())

    @staticmethod
    def get_problem_type():
        problem_type = ProblemType.get(sys.argv["problem"])
        print("* Problem type: ", problem_type)
        return problem_type

    @staticmethod
    def get_mode():
        mode = ModeType.get(sys.argv["mode"])
        print("* Mode: ", mode)
        return mode

    @staticmethod
    def get_model_serialization_file_path():
        path = sys.argv["model"]
        print("* Serialize/Reuse trained model: ", path)
        return path

    @staticmethod
    def get_pipeline():
        pipeline = {{pipeline}}
        print("* Pipeline: ", pipeline)
        return pipeline

    @staticmethod
    def get_fit_data_file_path():
        path = sys.argv["fit"]
        print("* Fitting on data in file: ", path)
        return path

    @staticmethod
    def get_fit_output_file_path():
        path = sys.argv["fitOutput"]
        print("* Write output of fit to file: ", path)
        return path

    @staticmethod
    def get_predict_data_file_path():
        path = sys.argv["predict"]
        print("* Predicting on data in file: ", path)
        return path

    @staticmethod
    def get_predict_output_file_path():
        path = sys.argv["predictOutput"]
        print("* Write output of predict to file: ", path)
        return path

    @staticmethod
    def get_target_indices():
        target_indices = list(map(int, sys.argv["targets"].split()))
        print("* Target Indices: ", str(target_indices))
        return target_indices

    @staticmethod
    def get_seed():
        seed = sys.argv["seed"]
        print("* Seed: ", seed)
        return int(seed)


def execute():
    np.random.seed(ArgsHandler.get_seed())
    mode = ArgsHandler.get_mode()
    pipeline = ArgsHandler.get_pipeline()
    
    dataset_parser = DatasetParser()
    if mode is ModeType.FIT or mode is ModeType.FIT_AND_PREDICT:
        dataset_fit = dataset_parser.read_from_file(ArgsHandler.get_fit_data_file_path(), target_index=-1)
        pipeline.fit(X=dataset_fit, y=dataset_fit.target)

        if problem_type == ProblemType.TIME_SERIES_FEATURE_ENGINEERING:
            dataset_fit_transformed = pipeline.transform(dataset_fit)
            dataset_parser.write_to_file(dataset_fit_transformed, ArgsHandler.get_fit_output_file_path(), use_at_target_notation=False)

        model_serialization_file = ArgsHandler.get_model_serialization_file_path()
        if model_serialization_file is not None:
            dump(pipeline, model_serialization_file)

    if mode is ModeType.PREDICT:
        model_serialization_file = ArgsHandler.get_model_serialization_file_path()
        pipeline = load(model_serialization_file)

    if mode is ModeType.PREDICT or mode is ModeType.FIT_AND_PREDICT:
        dataset_predict = dataset_parser.read_from_file(ArgsHandler.get_predict_data_file_path(), target_index=-1)

        if problem_type == ProblemType.TIME_SERIES_FEATURE_ENGINEERING:
            dataset_predict_transformed = pipeline.transform(dataset_predict)
            dataset_parser.write_to_file(dataset_predict_transformed, ArgsHandler.get_predict_output_file_path(), use_at_target_notation=False)

        elif problem_type == ProblemType.TIME_SERIES_REGRESSION:
            predictions = pipeline.predict(dataset_predict)
            print("\tPredictions:" + str(predictions))
            
            # Make sure the predictions are in a list
            predictions = predictions.tolist() # TODO check if working
            
            # Convert possible integers to floats (nescassary for Weka signature)
            if isinstance(predictions[0], int):
                predictions = [float(i) for i in predictions]
            elif isinstance(predictions[0], list):
                for sublist in predictions:
                    sublist = [float(i) for i in sublist]
            if not isinstance(predictions[0], list):
                predictions = [predictions]
            predictions_json = json.dumps(predictions)
            
            predictions_file = ArgsHandler.get_predict_output_file_path()
            with open(predictions_file, 'w') as file:
                file.write(predictions_json)
    
    print("DONE.")


if __name__ == "__main__":
    print("CURRENT_PID:" + str(os.getpid()))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (4294967296, hard))

    ArgsHandler.setup()
    problem_type = ArgsHandler.get_problem_type()
    if problem_type == ProblemType.TIME_SERIES_FEATURE_ENGINEERING or problem_type == ProblemType.TIME_SERIES_REGRESSION:
        try:
            execute()
        except Exception as e:
            raise RuntimeError(f"Could not evaluate pipeline due to following reason: \n{e}")
    else:
        raise RuntimeError(f"Unsupported problem type: {ArgsHandler.get_problem_type()}")
