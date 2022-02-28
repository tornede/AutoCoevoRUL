import os
from ml4pdm.data import DatasetParser


def check_compatibility_of_datasets(directory):
	for file in os.listdir(directory):
		path = os.path.join(directory,file)
		if os.path.isdir(path):
			check_compatibility_of_datasets(path)
		elif file.endswith('.pdmff') or file.endswith('.arff'):
			try:
				DatasetParser().read_from_file(path)
				print(f"Parsing of file {path} successfull")
			except Exception as e:
				print(f"Parsing of file {path} failed with: \n\t{e}")


if __name__ == '__main__':
    check_compatibility_of_datasets('data')
    check_compatibility_of_datasets("/Users/tanja/Development/PdM/Publications/SensorSelection/AutoCoevoRUL/AutoCoevoRUL/tmp/tmp1/")
    