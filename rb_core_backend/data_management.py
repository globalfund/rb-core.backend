import json
import logging
import os
import re
import shutil

import pandas as pd

from rb_core_backend.util import profile_method

logger = logging.getLogger(__name__)
RAW_DATA_TYPES = {  # Rawgraphs has date, string or number, default to string when using
    "object": "string",
    "datetime64[ns]": {"type": "date", "dateFormat": "YYYY-MM-DD"},
    "int64": "number",
    "float64": "number",
}


class RBCoreDataManagement:
    """RB Core Data Management class for operations."""

    def __init__(self, location: str = None) -> None:
        """Initialize the data manager with a location.
        If no location is provided, use the environment variable DF_LOC.

        Args:
            location (str, optional): The location of the datasets. Defaults to None.

        Raises:
            ValueError: If no location is provided and the environment variable DF_LOC is not set.
        """
        print("INIT RB CORE DATA MANAGEMENT, LOCATION:", location)
        self.df_loc = location or os.environ.get("DF_LOC")
        if not self.df_loc:
            raise ValueError(
                "DF_LOC must be set either via parameter or environment variable."
            )

    @profile_method
    def create_parsed_file(self, df: pd.DataFrame, filename: str) -> None:
        """We prepare the data as JSON with the following properties:
        {
            "dataset": []
            "dataTypes": {
                "column1": "type",
                "column2": "type",
                ...
            }
            "errors": []
            "count": N
            "sample": []  # subset of the data
            "stats": []  # descriptive statistics for each column
        }

        Args:
            df (pandas.DataFrame): The dataframe to be converted
            filename (str): The name of the file to be created
        """
        logger.debug("Creating parsed file")
        loc = f"{self.df_loc}parsed-data-files/{filename}.json"
        sample_loc = f"{self.df_loc}sample-data-files/{filename}.json"

        # 1. Map dtypes safely
        data_types = {
            col: RAW_DATA_TYPES.get(dtype.name, "string")
            for col, dtype in df.dtypes.items()
        }

        # 2. Format datetimes to 'YYYY-MM-DD'
        for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
            df[col] = df[col].dt.strftime("%Y-%m-%d")

        # 3. Replace NaN with None for JSON serialization and export
        df = df.where(pd.notnull(df), None)

        # 4. Convert to list of dicts
        cleaned_data = df.to_dict(orient="records")

        # Ensure "Year" columns, which are '%Y-%m-%d', where %m and %d are stored only as the year itself
        cleaned_data = self._convert_single_year_dates(cleaned_data)
        stats = self._get_dataset_stats(df)
        # save parsed at loc
        self._json_save(
            {
                "dataset": cleaned_data,
                "dataTypes": data_types,
                "errors": [],
                # Also include the sample data in the parsed file in case it is useful
                "count": len(cleaned_data),
                "sample": cleaned_data[:10],
                "stats": stats,
            },
            loc,
        )

        self._json_save(
            {
                "dataset": cleaned_data[:10],
                "dataTypes": data_types,
                "errors": [],
                "count": len(cleaned_data),
                "stats": stats,
            },
            sample_loc,
            indent=4,
        )
        # save the first 10 items to the sample data file
        with open(sample_loc, "w") as f:
            json.dump(
                {
                    "dataset": cleaned_data[:10],
                    "dataTypes": data_types,
                    "errors": [],
                    "count": len(cleaned_data),
                    "stats": stats,
                },
                f,
                indent=4,
            )

    @profile_method
    def remove_parsed_files(self, ds_name: str) -> str:
        """Removing parsed files

        Args:
            ds_name (str): The name of the dataset

        Returns:
            str: Success or error message
        """
        logger.debug("Removing parsed files")
        try:
            parsed_df = f"{self.df_loc}parsed-data-files/{ds_name}.json"
            sample_df = f"{self.df_loc}sample-data-files/{ds_name}.json"
            # remove the parsed files if they exist
            if os.path.exists(parsed_df):
                os.remove(parsed_df)
            if os.path.exists(sample_df):
                os.remove(sample_df)
            return "Success"
        except Exception as e:
            logger.error(f"Error in remove_parsed_files: {str(e)}")
            return "Sorry, something went wrong in our update. Contact us for more information."

    @profile_method
    def duplicate_parsed_files(self, ds_name: str, new_ds_name: str) -> str:
        """Duplicate parsed files

        Args:
            ds_name (str): The name of the dataset
            new_ds_name (str): The name of the new dataset

        Return:
            str: Success or error message
        """
        logger.debug("Duplicating specified parsed files")
        try:
            parsed_dataset = f"{self.df_loc}parsed-data-files/{ds_name}.json"
            sample_dataset = f"{self.df_loc}sample-data-files/{ds_name}.json"
            new_parsed_dataset = f"{self.df_loc}parsed-data-files/{new_ds_name}.json"
            new_sample_dataset = f"{self.df_loc}sample-data-files/{new_ds_name}.json"
            # duplicate the parsed files if they exist
            if os.path.exists(parsed_dataset):
                shutil.copy(parsed_dataset, new_parsed_dataset)
            if os.path.exists(sample_dataset):
                shutil.copy(sample_dataset, new_sample_dataset)
            return "Success"
        except Exception as e:
            logger.error(f"Error in duplicate_parsed_files: {str(e)}")
            return "Sorry, something went wrong in our duplication. Contact us for more information."

    @profile_method
    def load_sample_data(self, dataset_id: str) -> dict:
        """Read and return the sample data for a given dataset id in the form required by the frontend.

        Args:
            dataset_id (str): The id of the dataset

        Returns:
            dict: A dictionary containing the sample data
        """
        try:
            logger.debug("Sampling data")
            loc = f"{self.df_loc}sample-data-files/{dataset_id}.json"
            try:
                with open(loc, "r") as f:
                    data = json.load(f)
            except Exception:
                return "Sorry, this dataset is not available. Please contact us for more information."
            res = {
                "count": data["count"],
                "dataTypes": data["dataTypes"],
                "sample": data["dataset"][:10],
                "filterOptionGroups": list(data["dataTypes"].keys()),
                "stats": data["stats"],
            }

            return res
        except Exception as e:
            logger.error(f"Error in load_sample_data: {str(e)}")
            return "Sorry, we could not read the data from the provided dataset. Contact us for more information."

    @profile_method
    def load_parsed_data(
        self, dataset_id: str, page: int = 1, page_size: int = 10
    ) -> dict:
        """Read and return the parsed data for a given dataset id in the form required by the frontend.
        Paginated with page and page_size.

        Args:
            dataset_id (str): The id of the dataset
            page (int): The page number
            page_size (int): The number of items per page

        Returns:
            dict: A dictionary containing the parsed data
        """
        try:
            logger.debug("Loading parsed data, paginated")
            loc = f"{self.df_loc}parsed-data-files/{dataset_id}.json"
            with open(loc, "r") as f:
                data = json.load(f)

            start = (page - 1) * page_size
            end = start + page_size
            res = {"count": data["count"], "data": data["dataset"][start:end]}

            return res
        except Exception as e:
            logger.error(f"Error in load_parsed_data: {str(e)}")
            return "Sorry, we were unable to retrieve the data for this dataset. Contact us for more information."

    @profile_method
    def get_dataset_size(self, dataset_ids: list[str]) -> float:
        """Compute the size in MB for datasets

        Args:
            dataset_ids (list[str]): The ids of the datasets

        Returns:
            float: The total size in MB of the datasets
        """
        try:
            total_size_in_mb = 0
            logger.debug("Computing dataset sizes")

            for dataset_id in dataset_ids:
                loc = f"{self.df_loc}parsed-data-files/{dataset_id}.json"
                try:
                    file_stats = os.stat(loc)
                    total_size_in_mb += file_stats.st_size / (1024 * 1024)
                except FileNotFoundError:
                    continue
            return total_size_in_mb
        except Exception as e:
            logger.error(f"Error in get_dataset_size: {str(e)}")
            return "Sorry, we were unable to compute the sizes for the dataset(s). Contact us for more information."

    @staticmethod
    @profile_method
    def _json_save(data: dict, loc: str, indent: int = None) -> None:
        """Save data as JSON to the specified location.

        Args:
            data (dict): The prepared dict to be saved
            loc (str): The path to save the JSON file
            indent (int, optional): An indentation to pretty print the JSON. Defaults to None.
        """
        with open(loc, "w") as f:
            if not indent:
                json.dump(data, f)
            else:
                json.dump(data, f, indent=indent)

    @staticmethod
    @profile_method
    def _get_dataset_stats(df: pd.DataFrame) -> list[dict]:
        """Generate descriptive statistics for each column of the DataFrame.

        This function processes the provided DataFrame and generates statistical summaries for each column.
        Depending on the nature and distribution of data, it categorizes the statistics into 'percentage' for
        common categories, 'bar' for moderate unique values, and 'unique' for columns with many unique values.

        Args:
            df (pandas.DataFrame): The DataFrame to be processed

        Return:
            list[dict]: A list of dictionaries containing the column name, type of statistics, and the data
        """
        try:
            stats = []

            for c in df.columns:
                unique_values = df[c].nunique()

                if unique_values < 4 or (
                    unique_values < len(df) / 1.5 and unique_values > 20
                ):
                    data = df[c].value_counts(normalize=True).reset_index()
                    data.columns = ["name", "value"]
                    data["value"] = data["value"] * 100
                    data = data.sort_values(by="value", ascending=False)

                    if len(data) > 20:
                        others_value = data.iloc[2:]["value"].sum()
                        others_data = pd.DataFrame(
                            [{"name": "Others", "value": others_value}]
                        )
                        data = pd.concat(
                            [data.iloc[:2], others_data], ignore_index=True
                        )

                    stats.append(
                        {
                            "name": c,
                            "type": "percentage",
                            "data": data.to_dict(orient="records"),
                        }
                    )

                elif unique_values < 21:
                    data = df[c].value_counts().reset_index()
                    data.columns = ["name", "value"]
                    data = data.sort_values(by="name")
                    stats.append(
                        {
                            "name": c,
                            "type": "bar",
                            "data": data.to_dict(orient="records"),
                        }
                    )

                else:
                    stats.append(
                        {
                            "name": c,
                            "type": "unique",
                            "data": [{"name": "Unique", "value": unique_values}],
                        }
                    )

            return stats
        except Exception as e:
            logger.error("Error in get_dataset_stats: " + str(e))
            return "Error"

    @staticmethod
    @profile_method
    def _convert_single_year_dates(cleaned_data: dict) -> dict:
        """Takes an array of cleaned data dicts, and converts any date strings

        Args:
            cleaned_data (dict): the input data.

        Returns:
            dict: the data with any dates converted.
        """
        # collapse %Y-%m-%d -> %Y if month/day are always 01
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

        # find all keys across cleaned_data
        all_keys = {k for row in cleaned_data for k in row.keys()}

        for col in all_keys:
            # collect non-null string values for this column
            str_values = [
                row[col]
                for row in cleaned_data
                if col in row and isinstance(row[col], str)
            ]

            if not str_values:
                continue

            # check if all are YYYY-MM-DD
            if all(date_pattern.match(v) for v in str_values):
                # check if all are YYYY-01-01
                if all(v.endswith("-01-01") for v in str_values):
                    # collapse to YYYY
                    for row in cleaned_data:
                        if col in row and isinstance(row[col], str):
                            row[col] = row[col][:4]
        return cleaned_data
