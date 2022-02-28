import os
from ml4pdm.data import DatasetParser


def check_compatibility_of_datasets():
	directory = 'data'
	for folder in os.listdir(directory):
		if os.path.isdir(os.path.join(directory, folder)):
			print(f"Now iterating over files in folder {os.path.join(directory, folder)}")
			for file in os.listdir(os.path.join(directory, folder)):
				try:
					DatasetParser().read_from_file(os.path.join(directory, folder, file))
				except ValueError as e:
					print(f"Parsing failed with: \n{e}")
				print(f"Parsing of file {os.path.join(directory, folder, file)} successfull")


if __name__ == '__main__':
    check_compatibility_of_datasets()