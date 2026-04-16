""" 
test script to validate the dataset

checks csv schema, image file existance, and smiles validity 
"""

from src.validate_data import validate_data
from test.utils import resolve_dataset_paths


image_dir, label_path = resolve_dataset_paths()

overall_ok, results = validate_data(image_dir, label_path)

print("Overall Valid:", overall_ok)

for category, (status, issues) in results.items():
    print(f"\n{category.upper()} VALID:", status)

    if issues:
        for issue in issues:
            print(issue)
