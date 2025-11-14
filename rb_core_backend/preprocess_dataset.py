import base64
import datetime
import logging
import os
import re
import time
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from rb_core_backend.data_management import RBCoreDataManagement
from rb_core_backend.read_data import RBCoreDataAPIReader, RBCoreDataFileReader
from rb_core_backend.util import DATE_FORMATS, configure_logger, profile_method


class PreprocessDataOptions:
    """Options for preprocessing data, with default values set to True for all options.
    Allows the user to skip certain preprocessing steps if desired.
    """

    def __init__(
        self,
        strip_metadata: bool = True,
        drop_empty_rows: bool = True,
        drop_empty_columns: bool = True,
        clean_headers: bool = True,
        convert_dates: bool = True,
        fill_na: bool = True,
        column_percentages: bool = True,
        column_number_string_to_numeric: bool = True,
        column_comma_numeric: bool = True,
        column_convert_numeric_strings: bool = True,
    ) -> None:
        self.strip_metadata = strip_metadata
        self.drop_empty_rows = drop_empty_rows
        self.drop_empty_columns = drop_empty_columns
        self.clean_headers = clean_headers
        self.convert_dates = convert_dates
        self.fill_na = fill_na
        self.column_percentages = column_percentages
        self.column_number_string_to_numeric = column_number_string_to_numeric
        self.column_comma_numeric = column_comma_numeric
        self.column_convert_numeric_strings = column_convert_numeric_strings


class RBCoreDatasetPreprocessor(ABC):
    """Dataset Preprocessor class for preprocessing datasets."""

    def __init__(
        self,
        data_manager: RBCoreDataManagement,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        """Initialize the RBCoreDatasetPreprocessor class with a logger and data manager.

        Args:
            data_manager (RBCoreDataManagement, optional): data manager instance. Defaults to a new instance.
            logger (logging.Logger, optional): Provided logger. Defaults to logging.getLogger(__name__).
        """
        self.logger = logger
        self.data_manager = data_manager

    @profile_method
    @abstractmethod
    def preprocess_data(
        self,
        name: str,
        create_ds: bool = False,
        table: str = None,
        db: dict = None,
        api: dict = None,
        options: PreprocessDataOptions = PreprocessDataOptions(),
    ) -> str:
        """Process trigger to preprocess a dataset.
        The headers are updated to include the dataset name and only have a-zA-Z0-9 values.
        We make sure each column is the correct dtype and has no NaN values.
        Lastly we make sure the first row contains the correct dtype in each column, to enforce solr indexing.
        This process saves the preprocessed dataset by overwriting the provided file.

        Can be overwritten for custom preprocessing steps, or used with super().preprocess_data.

        Args:
            name (str): name of the dataset to preprocess
            create_ds (bool, optional): boolean to indicate creation of dataset storage. Defaults to False.
            table (str, optional): table name if reading from a database. Defaults to None.
            db (dict, optional): database connection info if reading from a database. Defaults to None.
            api (dict, optional): api connection info if reading from an api. Defaults to None.
            options (PreprocessDataOptions, optional): options for preprocessing steps. Defaults to true for all.

        Returns:
            str: success message if the dataset was preprocessed successfully, error message otherwise
        """
        self.logger.debug(f"Preprocessing data for {name}")
        file_path = f"./staging/{name}" if not db else name
        res = "Success"

        # Get the extension length
        try:
            filename, _ = os.path.splitext(name)
        except Exception:
            return "Unable to process the file, no file extension provided."

        # Read and process the data
        try:
            self.logger.debug("-- Preprocessing content")
            # Read the data into self.df
            message = self._read_data(file_path, table, db, api)
            self.logger.debug(f"---- Reading data result: {message}")
            if "Success" not in message:
                return message

            # Early type inference for better performance
            self._infer_objects()

            # Strip metadata from input files.
            if options.strip_metadata:
                self._strip_metadata()

            if options.drop_empty_rows:
                # drop any row that is 90-100% empty
                self._drop_empty_rows()

            if options.drop_empty_columns:
                # drop any column that is 95-100% empty
                self._drop_empty_cols()

            if options.clean_headers:
                # Clean the headers to only a-z, A-Z, 0-9
                self.df.columns = self.df.columns.astype(str)
                self.df.columns = self.df.columns.str.replace(
                    r"[^a-zA-Z0-9%]", "", regex=True, flags=re.IGNORECASE
                )

            if options.convert_dates:
                # Process dates
                self._check_and_convert_dates()

            self._preprocess_columns(options)

            # prep by cleaning the dataframe from any NA values
            if options.fill_na:
                self._fillna_on_dtype()

            self.logger.debug(f"Dataframe:\n{self.df.head()}")
            self.logger.debug(f"Done preprocessing data for {name}")
            if create_ds:
                self.data_manager.create_parsed_file(self.df, filename)
        except Exception as e:
            self.logger.error(f"Error in preprocess_data: {str(e)}")
            res = "Sorry, something went wrong in our dataset processing. Contact the admin for more information."
        return res

    @profile_method
    def _preprocess_columns(self, options: PreprocessDataOptions) -> None:
        """Single pass through columns with vectorized operations.

        Args:
            options (PreprocessDataOptions): options for preprocessing steps.
        """
        columns_to_drop = []
        new_columns = {}

        # Pre-compute column dtypes to avoid repeated checks
        col_dtypes = self.df.dtypes

        for header in self.df.columns.tolist():
            col = self.df[header]
            col_len = len(col)
            non_na_count = col.notna().sum()

            # Skip empty columns
            if non_na_count == 0:
                continue

            # Only process object/string columns for text-based transformations
            if col_dtypes[header] == "object":
                # Check for percentage columns (vectorized)
                if options.column_percentages:
                    pct_result = self._process_percentage_column(col, col_len)
                    if pct_result is not None:
                        new_columns[header + "%"] = pct_result
                        columns_to_drop.append(header)
                        continue

                # Check for string values in numeric-like columns
                if options.column_number_string_to_numeric:
                    # Try converting to numeric - if mostly successful, it's numeric
                    numeric_test = pd.to_numeric(col, errors="coerce")
                    numeric_count = numeric_test.notna().sum()

                    # If less than 5% are non-numeric strings, clean them
                    if numeric_count >= non_na_count * 0.95:
                        # Remove non-numeric characters and convert
                        col = col.astype(str).str.replace(r"[^0-9.]", "", regex=True)
                        col = col.replace("", "0")
                        new_columns[header] = pd.to_numeric(
                            col, errors="coerce"
                        ).fillna(0)
                        continue

                # Check for comma-separated numbers (vectorized)
                if options.column_comma_numeric:
                    col = self._process_comma_numbers(col)
                # Try converting numeric strings to numbers
                if options.column_convert_numeric_strings:
                    numeric_converted = pd.to_numeric(col, errors="coerce")
                    numeric_ratio = numeric_converted.notna().sum() / non_na_count

                    if numeric_ratio >= 0.5:
                        new_columns[header] = numeric_converted
                        continue

            # Store the processed column if it was modified
            if header not in columns_to_drop and header not in new_columns:
                new_columns[header] = col

        # Batch update: drop old columns and add new ones
        if columns_to_drop:
            self.df = self.df.drop(columns=columns_to_drop)

        # Update with new/modified columns
        for col_name, col_data in new_columns.items():
            if col_name not in self.df.columns or not self.df[col_name].equals(
                col_data
            ):
                self.df[col_name] = col_data

    def _process_percentage_column(
        self, col: pd.Series, col_len: int
    ) -> pd.Series | None:
        """Vectorized percentage column processing.

        Args:
            col (pd.Series): column to process
            col_len (int): length of column

        Returns:
            pd.Series | None: processed column or None if not a percentage column
        """
        try:
            # Vectorized checks
            is_string = col.apply(lambda x: isinstance(x, str))
            if not is_string.any():
                return None

            ends_with_pct = col.str.endswith("%", na=False)
            if not ends_with_pct.any():
                return None

            # Extract numeric part (remove %)
            numeric_part = col.str[:-1]

            # Check if numeric
            numeric_values = pd.to_numeric(numeric_part, errors="coerce")
            valid_count = numeric_values.notna().sum()

            # If 75% or more are valid percentages, convert
            if valid_count / col_len > 0.75:
                return numeric_values

            return None
        except Exception as e:
            self.logger.error(f"Error in _process_percentage_column: {str(e)}")
            return None

    def _process_comma_numbers(self, col: pd.Series) -> pd.Series:
        """Vectorized comma number processing (handles both 1,234.56 and 1.234,56 formats).

        Args:
            col (pd.Series): column to process

        Returns:
            pd.Series: processed column
        """
        try:
            if col.dtype != "object":
                return col

            # if col.str contains any alphabetic characters, return col
            has_alpha = col.str.contains(r"[a-zA-Z]", na=False, regex=True)
            if has_alpha.any():
                return col

            # Check which format is being used
            has_comma = col.str.contains(",", na=False)
            has_dot = col.str.contains(r"\.", na=False, regex=True)

            if not has_comma.any():
                return col

            # For values with both comma and dot, assume 1,234.56 format (remove comma)
            # For values with only comma, assume 1.234,56 format (replace comma with dot)
            both_mask = has_comma & has_dot
            comma_only_mask = has_comma & ~has_dot

            if both_mask.any():
                col = col.copy()  # Add this line
                col.loc[both_mask] = col.loc[both_mask].str.replace(
                    ",", "", regex=False
                )

            if comma_only_mask.any():
                if (
                    "copy" not in locals() or not col._is_copy
                ):  # Only copy if not already copied
                    col = col.copy()
                col.loc[comma_only_mask] = col.loc[comma_only_mask].str.replace(
                    ",", ".", regex=False
                )

            return col
        except Exception as e:
            self.logger.error(f"Error in _process_comma_numbers: {str(e)}")
            return col

    @profile_method
    def _check_and_convert_dates(self, date_formats=DATE_FORMATS) -> pd.DataFrame:
        """Date conversion with early filtering and pandas built-in inference.

        Args:
            date_formats (list, optional): list of date formats to check against. Defaults to DATE_FORMATS.

        Returns:
            pd.DataFrame: dataframe with the dates converted to a standard format
        """
        # Only check object/string columns (skip numeric columns entirely)
        string_cols = self.df.select_dtypes(include=["object"]).columns

        for header in string_cols:
            col = self.df[header].dropna()

            if len(col) == 0:
                continue

            # Get first non-null value (no need to sort)
            first_value = str(col.iloc[0])

            # Try pandas built-in date inference first (fastest)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.",  # NOQA: E501
                        category=UserWarning,
                    )
                    converted = pd.to_datetime(col, errors="coerce")
                    success_rate = converted.notna().sum() / len(col)

                    if success_rate > 0.8:  # 80% successful conversion
                        # Check if all times are midnight (00:00:00)
                        if converted.dt.time.nunique() == 1 and converted.dt.time.iloc[
                            0
                        ] == datetime.time(0, 0):
                            self.df[header] = converted.dt.normalize()
                        else:
                            self.df[header] = converted
                        continue
            except Exception:
                pass

            # Check if it's a year column (fast check)
            if self._is_year_column(col):
                self.df[header] = pd.to_datetime(col, format="%Y", errors="coerce")
                continue

            # Fallback: try custom date formats
            presumed_dateformat = None
            has_time = False

            for date_format in date_formats:
                try:
                    datetime.datetime.strptime(first_value, date_format)
                    presumed_dateformat = date_format
                    has_time = "H" in date_format or "I" in date_format
                    break
                except ValueError:
                    continue

            if presumed_dateformat is None:
                continue

            # Convert with the found format
            try:
                if has_time:
                    converted = pd.to_datetime(
                        col, format=presumed_dateformat, errors="coerce"
                    )
                    # Check if all times are midnight
                    if (converted.dt.time == datetime.time(0, 0)).all():
                        self.df[header] = converted.dt.normalize()
                    else:
                        self.df[header] = converted
                else:
                    self.df[header] = pd.to_datetime(
                        col, format=presumed_dateformat, errors="coerce"
                    )
            except Exception as e:
                self.logger.error(
                    f"Error converting dates in column {header}: {str(e)}"
                )
                continue

    @profile_method
    def _fillna_on_dtype(self) -> None:
        """Fill NaN values in batches by dtype."""
        fill_values = {
            "object": "",
            "int64": 0,
            "float64": 0.0,
            "datetime64[ns]": pd.Timestamp("1970-01-01"),
        }

        # Group columns by dtype and fill in batches
        for dtype_str, fill_value in fill_values.items():
            cols = self.df.select_dtypes(include=[dtype_str]).columns
            if len(cols) > 0:
                self.df[cols] = self.df[cols].fillna(fill_value)

    @profile_method
    def _find_almost_empty_row_index(self, threshold=0.9) -> int:
        """Get the index of rows that are nearly empty.

        Args:
            threshold (float, optional): _description_. Defaults to 0.9.

        Returns:
            int | pd.Null: the first index of an almost empty row as defined by the threshold or pd.NA if none found
        """
        # Calculate the percentage of NaN values in each row
        nan_percentage = self.df.isnull().mean(axis=1)
        # Find rows where the percentage of NaN values exceeds the threshold
        almost_empty_rows = self.df[nan_percentage >= threshold]
        # If there are no almost empty rows, return NaN
        if almost_empty_rows.empty:
            return pd.NA
        # Return the index of the first almost empty row
        return almost_empty_rows.index.min()

    @profile_method
    def _remove_rows_after_empty(self) -> None:
        """Find and drop the index of the first completely empty row"""
        empty_row_index = self._find_almost_empty_row_index()
        if not pd.isnull(empty_row_index):
            # Drop rows after the empty row
            self.df = self.df.iloc[:empty_row_index]

    @profile_method
    def _strip_metadata(self) -> None:
        """Function to strip metadata and empty rows from the dataframe.
        This function removes any rows and columns that are completely empty (all NaN values).
        """
        try:
            # In the first row, if every value that is not NaN is a string starting with a #, drop that row
            # Catches cases such as HXL metadata rows
            first_row = self.df.iloc[0]
            if (
                first_row.dropna()
                .apply(lambda x: isinstance(x, str) and x.strip().startswith("#"))
                .all()
            ):
                self.df = self.df.iloc[1:].reset_index(drop=True)

            self._remove_rows_after_empty()  # Drop bottom comments or metadata.

            # Drop rows and columns with all NaN values
            self.df = self.df.dropna(axis=0, how="all").dropna(axis=1, how="all")

            if self.df.empty:
                return

            # Vectorized approach: find the bounding box of non-NaN values
            # Create a boolean mask of non-NaN values
            non_nan_mask = self.df.notna()

            # Find columns that have at least one non-NaN value
            cols_with_data = non_nan_mask.any(axis=0)
            if not cols_with_data.any():
                return

            # Get the first and last column indices with data
            col_indices = np.where(cols_with_data)[0]
            min_col_idx = col_indices[0]
            max_col_idx = col_indices[-1]

            # Slice the dataframe to keep only columns with data
            self.df = self.df.iloc[:, min_col_idx:max_col_idx + 1]

            first_data_idx = self._find_first_data_row()
            self.df = self.df.iloc[first_data_idx:].reset_index(drop=True)
        except Exception as e:
            self.logger.error(f"Error in strip_metadata: {str(e)}")

    @profile_method
    def _find_first_data_row(self, min_non_na_ratio=0.6) -> int:
        """Find the index of the first row with a minimum ratio of non-NA values.

        Args:
            min_non_na_ratio (float, optional): ratio determining min content. Defaults to 0.6.

        Returns:
            int: row nr of the first row with enough non-NA values
        """
        for i, row in self.df.iterrows():
            non_na_ratio = row.count() / len(self.df.columns)
            if non_na_ratio >= min_non_na_ratio:
                return i
        return 0

    @profile_method
    def _read_data(self, file_path, table, db, api) -> str:
        """Read the incoming data from file, database, or api.

        Args:
            file_path (str): path to the file to read
            table (str): table name if reading from a database
            db (dict): database connection info if reading from a database
            api (dict): api connection info if reading from an api

        Returns:
            str: success message if the data was read successfully, error message otherwise
        """
        if table:
            res = RBCoreDataFileReader(
                file_path, optional_args={"table": table}
            ).read_data()
        elif db:
            res = RBCoreDataFileReader(file_path, optional_args=db).read_data()
        elif api:
            # replace _ with / in api_url
            url = base64.b64decode(api["api_url"].replace("_", "/"))
            additional_args = {}
            if api.get("json_root", None) != "none":
                additional_args["json_root"] = api["json_root"]
            if api.get("xml_root", None) != "none":
                additional_args["xml_root"] = api["xml_root"]

            res = RBCoreDataAPIReader(
                url, additional_args=additional_args
            ).read_data_from_api()
        else:
            res = RBCoreDataFileReader(file_path).read_data()

        if isinstance(res, tuple):
            df, message = res
        else:
            df = res
            message = "We were unable to parse your data into a dataframe. Contact our service team for assistance."

        if type(df) is not pd.DataFrame:
            self.logger.error(f"Error in preprocess_data: {message}")

        # Store the DataFrame
        self.df = df
        return message

    @profile_method
    def _infer_objects(self) -> None:
        """Infer object types in the dataframe to optimize dtypes."""
        try:
            self.df = self.df.infer_objects()
        except Exception as e:
            self.logger.error(f"Error in _infer_objects: {str(e)}")

    @profile_method
    def _drop_empty_rows(self) -> None:
        """Drop rows that are completely empty (all NaN values)."""
        try:
            self.df = self.df.dropna(axis=0, thresh=self.df.shape[1] * 0.1)
        except Exception as e:
            self.logger.error(f"Error in _drop_empty_rows: {str(e)}")

    @profile_method
    def _drop_empty_cols(self) -> None:
        """Drop columns that are completely empty (all NaN values)."""
        try:
            self.df = self.df.dropna(axis=1, thresh=self.df.shape[0] * 0.05)
        except Exception as e:
            self.logger.error(f"Error in _drop_empty_cols: {str(e)}")

    @staticmethod
    def _is_year_column(series: pd.Series, start: int = 1900, end: int = 2100) -> bool:
        """Check if all non-null values in a pandas Series are integers
        representing years within [start, end].
        """
        # Drop missing values
        values = series.dropna()

        if len(values) == 0:
            return False

        # Try converting to numeric (will fail gracefully for non-numeric values)
        try:
            numeric_values = pd.to_numeric(values, errors="coerce")
        except Exception:
            return False

        # If any conversion failed, it's not a year column
        if numeric_values.isna().any():
            return False

        # Check range
        return numeric_values.between(start, end).all()


if __name__ == "__main__":
    """Not normally accessed as main, but useful for profiling preprocessing performance

    Create a custom preprocessor class that inherits from RBCoreDatasetPreprocessor to allow for all
    preprocessing steps, without any custom additions.

    Run it for all .csv files in the ./staging directory, measuring time taken and throughput.
    """
    configure_logger()

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

    print("=" * 80)
    print("PROFILING DATA PREPROCESSING")
    print("=" * 80)

    # for every filename .csv file in ./staging
    for filename in os.listdir("./staging"):
        if not filename.endswith(".csv"):
            continue

        print(f"\n{'=' * 80}")
        print(f"Processing: {filename}")
        print(f"{'=' * 80}")

        start = time.perf_counter()
        file_size_mb = os.path.getsize("./staging/" + filename) / (1024 * 1024)

        result = preprocessor.preprocess_data(
            name=filename,
            create_ds=True,
            options=options,
        )

        total_time = time.perf_counter() - start

        print(f"\n{'=' * 80}")
        print("📊 SUMMARY:")
        print("=" * 80)
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Total time: {total_time:.4f}s")
        print(f"  Result: {result}")
        print(f"  Throughput: {file_size_mb/total_time:.2f} MB/s")
        print("=" * 80)
        break
