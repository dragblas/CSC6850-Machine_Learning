"""
utility funcitons for test scripts
"""

import sys
import os

def resolve_dataset_paths():
	"""
	parses command line argument and returns: (image_dir, label_path)
	"""

	if len(sys.argv) != 2:
		print("Usage: python test/test_validate_data.py <dataset_name>")
		sys.exit(1)

	dataset_name = sys.argv[1]

	base_path = os.path.join("data", dataset_name)
	image_dir = os.path.join(base_path, "images")
	label_path = os.path.join(base_path, "labels.csv")

	return image_dir, label_path