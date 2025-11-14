# Dev

- [Dev](#dev)
  - [rb_core_backend](#rb_core_backend)
    - [rb_core_backend.external\_sources](#rb_core_backendexternal_sources)
      - [rb_core_backend.external\_sources.index](#rb_core_backendexternal_sourcesindex)
      - [rb_core_backend.external\_sources.kaggle](#rb_core_backendexternal_sourceskaggle)
      - [rb_core_backend.external\_sources.util](#rb_core_backendexternal_sourcesutil)
    - [rb_core_backend.data\_management](#rb_core_backenddata_management)
    - [rb_core_backend.mongo](#rb_core_backendmongo)
    - [rb_core_backend.preprocess\_dataset](#rb_core_backendpreprocess_dataset)
    - [rb_core_backend.read\_data](#rb_core_backendread_data)
    - [rb_core_backend.util](#rb_core_backendutil)
  - [Development directories](#development-directories)

## rb_core_backend

### rb_core_backend.external_sources

#### rb_core_backend.external_sources.index

A class RBCoreExternalSources, where all_sources is part of the class, which the user can then control. Used in app for api calls.

Expects the caller to create a RBCoreMongoClient, for example:

```python
mongo_client = RBCoreBackendMongo(
    mongo_host=os.getenv("MONGO_HOST", "localhost:27017"),
    mongo_username=os.getenv("MONGO_USERNAME", ""),
    mongo_password=os.getenv("MONGO_PASSWORD", ""),
    mongo_auth_source=os.getenv("MONGO_AUTH_SOURCE", ""),
    database_name=os.getenv("DATABASE_NAME", ""),
    fs_db_name=os.getenv("MONGO_FSDB", ""),
)
```

To use, create an instance of RBCoreExternalSources with the RBCoreBackendMongo instance, and optionally a dict of sources.

The dict of sources functions as follows:

```python

sources_dict = {
    "Source Name": {
        "download": download_function_for_source_name,
        "index": index_function_for_source_name,
    }
}
```

#### rb_core_backend.external_sources.kaggle

Contains a class `RBCoreExternalSourceKaggle` for external source data connection with Kaggle.

#### rb_core_backend.external_sources.util

Constants:

- Definition of the external dataset
- Definition of the external dataset resource
- Legacy format for external datasets

### rb_core_backend.data_management

Formerly "SSR". RBCoreDataManagement. Initialise with a location, if not provided checks the environment for `DF_LOC`.

Used for API calls:

- remove_parsed_files
- duplicate_parsed_files
- load_sample_data
- load_parsed_data
- get_dataset_size

Used in preprocess_dataset:

- create_parsed_file

### rb_core_backend.mongo

Used in rb_core_backend.external_sources

### rb_core_backend.preprocess_dataset

For usage, first initialise a DatasetProcessor class with parent, and RBCoreDataManagement with a location.

preprocess_data, used in:

- API calls.
- external_sources for the \_download functions

Example usage:

```python
class MyRBCoreDatasetPreprocessor(RBCoreDatasetPreprocessor):
    def preprocess_data(
        self,
        name: str,
        create_ds: bool = False,
        table: str = None,
        db: dict = None,
        api: dict = None,
        options: PreprocessDataOptions = PreprocessDataOptions(),
    ) -> str:
        # Custom preprocessing steps can be added here
        return super().preprocess_data(name, create_ds, table, db, api, options)

data_manager = RBCoreDataManagement("./")
preprocessor = MyRBCoreDatasetPreprocessor(data_manager=data_manager)
options = PreprocessDataOptions()
filename = "example.csv"
result = preprocessor.preprocess_data(
    name=filename,
    create_ds=True,
    options=options,
)
```

### rb_core_backend.read_data

Used in preprocess_dataset.

Split into two classes, `RBCoreDataFileReader` and `RBCoreDataAPIReader`.

### rb_core_backend.util

- configure_logger: to be used in the startup of the application
- profile_method: dev tool to measure time (and expandable for memory, etc.) per function.
- json_return: simple function to return a tuple: a dict with code, result, message, and a code. Used in API calls.
- functions to set up the ds locations

## Development directories

The following directories are included, but empty, for development purposes: `dev`, `logging`, `parsed-data-files`, `sample-data-files`, `staging`.
