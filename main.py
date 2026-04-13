import argparse
import json
import os
import shutil

from rb_core_backend.data_management import RBCoreDataManagement
from rb_core_backend.preprocess_dataset import PreprocessDataOptions, RBCoreDatasetPreprocessor
from rb_core_backend.util import configure_logger


class SimplePreprocessor(RBCoreDatasetPreprocessor):
    def preprocess_data(self, name, create_ds=False, table=None, db=None, api=None, options=PreprocessDataOptions()):
        return super().preprocess_data(name, create_ds, table, db, api, options)


def print_usage() -> None:
    print("rb-core-backend dev runner")
    print("")
    print("Commands:")
    print("  --run-preprocess-data FILE   Run the preprocessing flow on a local file")
    print("  --run-load-sample-data ID    Print the sample data for a dataset ID")
    print("  --run-load-parsed-data ID    Print a page of parsed data for a dataset ID")
    print("")
    print("Options:")
    print("  --no-create-ds               Skip writing parsed/sample output files")
    print("  --page N                     Page number for --run-load-parsed-data (default: 1)")
    print("  --page-size N                Page size for --run-load-parsed-data (default: 10)")
    print("")
    print("Examples:")
    print("  uv run main.py --run-preprocess-data data.csv")
    print("  uv run main.py --run-preprocess-data data.parquet --no-create-ds")
    print("  uv run main.py --run-load-sample-data main")
    print("  uv run main.py --run-load-parsed-data main --page 2 --page-size 5")


def run_preprocess_data(file: str, create_ds: bool = True) -> None:
    if not os.path.isfile(file):
        print(f"Error: file not found: {file}")
        raise SystemExit(1)

    for d in ("staging", "parsed-data-files", "sample-data-files", "logging"):
        os.makedirs(d, exist_ok=True)

    dx_name = os.path.basename(file)
    shutil.copy(file, f"staging/{dx_name}")

    configure_logger()

    data_manager = RBCoreDataManagement("./")
    preprocessor = SimplePreprocessor(data_manager=data_manager)

    result = preprocessor.preprocess_data(dx_name, create_ds=create_ds)
    print("Result:", result)


def run_load_sample_data(dataset_id: str) -> None:
    data_manager = RBCoreDataManagement("./")
    result = data_manager.load_sample_data(dataset_id)
    print(json.dumps(result, indent=2, default=str))


def run_load_parsed_data(dataset_id: str, page: int = 1, page_size: int = 10) -> None:
    data_manager = RBCoreDataManagement("./")
    result = data_manager.load_parsed_data(dataset_id, page=page, page_size=page_size)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-preprocess-data", metavar="FILE")
    parser.add_argument("--run-load-sample-data", metavar="ID")
    parser.add_argument("--run-load-parsed-data", metavar="ID")
    parser.add_argument("--no-create-ds", action="store_true")
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=10)
    args = parser.parse_args()

    if args.run_preprocess_data:
        run_preprocess_data(args.run_preprocess_data, create_ds=not args.no_create_ds)
    elif args.run_load_sample_data:
        run_load_sample_data(args.run_load_sample_data)
    elif args.run_load_parsed_data:
        run_load_parsed_data(args.run_load_parsed_data, page=args.page, page_size=args.page_size)
    else:
        print_usage()
