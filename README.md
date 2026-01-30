# rb-core-backend

Core functionality of the Report Builder.

## Usage

We have not yet published this as a package to PyPi. To use this as a package in a python project:

```bash
uv add "rb-core-backend @ git+https://github.com/globalfund/rb-core.backend.git"
# or
pip install git+https://github.com/globalfund/rb-core.backend.git
```

Import and use as a regular package, for example:

```python
from rb_core_backend.preprocess_dataset import RBCoreDatasetPreprocessor, PreprocessDataOptions
from rb_core_backend.data_management import RBCoreDataManagement

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

filename = "your_file.csv"
result = preprocessor.preprocess_data(
    name=filename,
    create_ds=True,
    options=options,
)
```

## Development

For more details, read [docs/dev](./docs/dev.md).

There are notes on the included data reader at [docs/read_data_notes](./docs/read_data_notes.md)

### UV

Project is managed and run with UV.

- Install UV: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Create a virtual environment: `uv venv`.
- Install the dependencies: `uv pip install ".[dev]"`
- Run locally, for example the sample method in `rb_core_backend.preprocess_dataset`, with `uv run -m rb_core_backend.preprocess_dataset`

### Usage of abstractmethod

There are several methods that are tagged with `@abstractmethod`. However, most of them contain default behaviour. In case you want to fully overwrite their content, you can by not referencing the method. A general example:

```python
from abc import ABC, abstractmethod
import pandas as pd
import logging

class BaseValidator(ABC):
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate parsed data"""
        if data.empty:
            raise ValueError("CSV file is empty")

        logger.info(f"CSV validation passed: {len(data)} rows, {len(data.columns)} columns")
        return True
```

and in a subclass:

```python
class CustomValidator(BaseValidator):
    def validate(self, data: pd.DataFrame) -> bool:
        # Optionally run base validation
        super().validate(data)

        # Add custom checks
        if "price" not in data.columns:
            raise ValueError("Missing price column")

        return True
```

### Usage of Kaggle

Install the kaggle CLI tool globally. `pip install -g kaggle` and set up your token as [per their guide](https://www.kaggle.com/docs/api)

For kaggle, as mentioned we need to set up the kaggle token.
Through docker, we copy it from the dx.backend root directory.
Make sure to download it set up your token as [from your account](https://www.kaggle.com/settings/account), and place it in the root folder.

### Code Management

*flake8* is used to maintain code quality in pep8 style

*isort* is used to maintain the imports

*pre-commit* is used to enforce commit styles in the form:

```bash
feat: A new feature
fix: A bug fix
docs: Documentation only changes
style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
refactor: A code change that neither fixes a bug nor adds a feature
perf: A code change that improves performance
test: Adding missing or correcting existing tests
chore: Changes to the build process or auxiliary tools and libraries such as documentation generation
```

### Commits

[Commitlint](https://github.com/conventional-changelog/commitlint#what-is-commitlint) is used to check your commit messages.

When setting up the repository, after locally setting up an environment, ensure pre-commit is installed:

- Add pre-commit: `uv add pre-commit` or `pip install pre-commit`.
- Install pre-commit to git: `uv run pre-commit install` or `pre-commit install`.
- Install the commit hook to git: `uv run pre-commit install --hook-type commit-msg` or `pre-commit install --hook-type commit-msg`.
