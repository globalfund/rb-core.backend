import copy
import datetime
import logging
import os
import re

import pandas as pd
import requests

from rb_core_backend.external_sources.model import ExternalSourceModel
from rb_core_backend.external_sources.util import EXTERNAL_DATASET_FORMAT, EXTERNAL_DATASET_RESOURCE_FORMAT
from rb_core_backend.mongo import RBCoreBackendMongo
from rb_core_backend.preprocess_dataset import RBCoreDatasetPreprocessor

logger = logging.getLogger(__name__)

WHO_SOURCE = "WHO"
# The WHO Global Health Observatory OData API: https://www.who.int/data/gho/info/gho-odata-api
WHO_API_URL = "https://ghoapi.azureedge.net/api"
WHO_INDICATOR_URL_PATTERN = re.compile(
    r"^https://ghoapi\.azureedge\.net/api/([A-Za-z0-9_.-]+)$"
)
WHO_SOURCE_NOTICE = (
    "  - This Datasource was retrieved from https://www.who.int/data/gho."
)
WHO_CATEGORY = "Health"
REQUEST_TIMEOUT = 180
# The number of dimension columns the GHO API exposes per observation (Dim1, Dim2, Dim3).
N_DIMENSIONS = 3
OBSERVATION_COLUMNS = [
    "SpatialDimType",
    "SpatialDim",
    "ParentLocation",
    "TimeDim",
    *[f"Dim{i}{suffix}" for i in range(1, N_DIMENSIONS + 1) for suffix in ("Type", "")],
    "NumericValue",
    "Low",
    "High",
    "Value",
    "Comments",
]


class RBCoreExternalSourceWHO(ExternalSourceModel):
    """RB Core External Source for the WHO Global Health Observatory, where every indicator is a dataset."""

    def __init__(
        self,
        mongo_client: RBCoreBackendMongo,
        dataset_preprocessor: RBCoreDatasetPreprocessor,
    ) -> None:
        super().__init__(mongo_client, dataset_preprocessor)

    def index(self, delete: bool = False) -> str:
        """Index all WHO GHO indicators into the federated search index.

        Args:
            delete (bool): Whether to remove the existing WHO entries before indexing.

        Returns:
            str: A string indicating the result of the indexing.
        """
        logger.info("WHO INDEX:: Indexing WHO data...")
        try:
            indicators = self._get(f"{WHO_API_URL}/Indicator")
        except Exception as e:
            logger.error(f"WHO INDEX:: Unable to retrieve the WHO indicators: {str(e)}")
            return "WHO - Unable to retrieve the indicators from the WHO API."

        if delete:
            logger.info("WHO INDEX:: - Removing old WHO data")
            self.mongo_client.mongo_remove_data_for_external_sources(WHO_SOURCE)
            existing_refs = set()
        else:
            existing_refs = {
                source.get("internalRef")
                for source in self.mongo_client.mongo_get_all_external_sources()
                if source.get("source") == WHO_SOURCE
            }

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        external_datasets = []
        for indicator in indicators:
            code = indicator.get("IndicatorCode")
            name = (indicator.get("IndicatorName") or "").strip()
            # Archived indicators only point at their replacement, they hold no data.
            if not code or not name or name.lower().startswith("archived") or code.upper().endswith("_ARCHIVED"):
                continue
            if code in existing_refs or not WHO_INDICATOR_URL_PATTERN.match(
                f"{WHO_API_URL}/{code}"
            ):
                continue
            external_datasets.append(
                self._create_external_source_object(code, name, now)
            )

        if not external_datasets:
            return "WHO - Successfully indexed 0 datasets."
        if self.mongo_client.mongo_create_external_sources(external_datasets) is None:
            return "WHO - Unable to store the indicators."
        logger.info("WHO INDEX:: WHO index updated successfully")
        return f"WHO - Successfully indexed {len(external_datasets)} datasets."

    def download(self, external_dataset: dict) -> str:
        """Download the observations of a WHO indicator, and preprocess them as a dataset.

        Args:
            external_dataset (dict): An external dataset object, of which the `url` is the indicator's API url
                and the `id` is the id to store the dataset under.

        Returns:
            str: "Success", or an error message.
        """
        match = WHO_INDICATOR_URL_PATTERN.match(external_dataset.get("url", ""))
        if not match:
            return "Sorry, we were unable to find the selected WHO dataset, please try a different dataset."
        code = match.group(1)
        ds_id = external_dataset.get("id", f"who-{code}")
        ds_name = f"{ds_id}.csv"
        staging_path = f"./staging/{ds_name}"
        logger.debug(f"WHO:: Downloading indicator {code} as {ds_name}")
        try:
            observations = self._get(f"{WHO_API_URL}/{code}")
            if not observations:
                return "Sorry, the selected WHO dataset does not contain any data, please try a different dataset."
            self._observations_to_dataframe(observations).to_csv(
                staging_path, index=False
            )
        except Exception as e:
            logger.error(f"WHO:: Unable to download indicator {code}: {str(e)}")
            return "Sorry, we were unable to download the selected WHO dataset, please try again later."

        try:
            return self.dataset_preprocessor.preprocess_data(ds_name, create_ds=True)
        finally:
            if os.path.exists(staging_path):
                os.remove(staging_path)

    @staticmethod
    def _get(url: str) -> list[dict]:
        """Request an OData collection from the GHO API, and return its values."""
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json().get("value", [])

    @staticmethod
    def _create_external_source_object(code: str, name: str, now: str) -> dict:
        """Create the federated search object for a WHO indicator."""
        uri = f"{WHO_API_URL}/{code}"
        external_dataset = copy.deepcopy(EXTERNAL_DATASET_FORMAT)
        external_dataset["title"] = name
        external_dataset["description"] = (
            f"WHO Global Health Observatory indicator {code}." + WHO_SOURCE_NOTICE
        )
        external_dataset["source"] = WHO_SOURCE
        external_dataset["URI"] = uri
        external_dataset["internalRef"] = code
        external_dataset["mainCategory"] = WHO_CATEGORY
        external_dataset["dateLastUpdated"] = now

        resource = EXTERNAL_DATASET_RESOURCE_FORMAT.copy()
        resource["title"] = name
        resource["description"] = name + WHO_SOURCE_NOTICE
        resource["URI"] = uri
        resource["internalRef"] = code
        resource["format"] = "json"
        resource["dateLastUpdated"] = now
        external_dataset["resources"].append(resource)
        return external_dataset

    def _observations_to_dataframe(self, observations: list[dict]) -> pd.DataFrame:
        """Convert GHO observations to a readable table: codes are replaced by their titles,
        and every dimension of the indicator (sex, age group, ...) becomes its own column.
        """
        df = pd.DataFrame(observations)
        df = df.reindex(columns=df.columns.union(OBSERVATION_COLUMNS, sort=False))
        titles_per_dimension = {}

        def titles(dimension: str) -> dict:
            if dimension not in titles_per_dimension:
                titles_per_dimension[dimension] = self._get_dimension_titles(dimension)
            return titles_per_dimension[dimension]

        def translate(dimension_types: pd.Series, codes: pd.Series) -> pd.Series:
            return pd.Series(
                [
                    (
                        titles(dimension).get(code, code)
                        if isinstance(dimension, str) and isinstance(code, str)
                        else code
                    )
                    for dimension, code in zip(dimension_types, codes)
                ],
                index=codes.index,
            )

        result = pd.DataFrame(index=df.index)
        result["Location"] = translate(df["SpatialDimType"], df["SpatialDim"])
        result["LocationCode"] = df["SpatialDim"]
        result["LocationType"] = df["SpatialDimType"].str.title()
        result["Region"] = df["ParentLocation"]
        result["Year"] = df["TimeDim"]

        for i in range(1, N_DIMENSIONS + 1):
            dimension_types = df[f"Dim{i}Type"].dropna().unique()
            for dimension in dimension_types:
                is_dimension = df[f"Dim{i}Type"] == dimension
                column = titles("__dimensions__").get(dimension, dimension.title())
                values = translate(df[f"Dim{i}Type"], df[f"Dim{i}"]).where(is_dimension)
                result[column] = (
                    values
                    if column not in result
                    else result[column].combine_first(values)
                )

        result["Value"] = df["NumericValue"]
        result["Low"] = df["Low"]
        result["High"] = df["High"]
        result["DisplayValue"] = df["Value"]
        result["Comments"] = df["Comments"]
        return result.dropna(axis=1, how="all")

    def _get_dimension_titles(self, dimension: str) -> dict:
        """Get the code to title mapping of a GHO dimension. Falls back to the codes themselves on failure."""
        try:
            if dimension == "__dimensions__":
                return {
                    item["Code"]: item["Title"]
                    for item in self._get(f"{WHO_API_URL}/Dimension")
                }
            if not re.match(r"^[A-Za-z0-9_]+$", dimension):
                return {}
            values = self._get(f"{WHO_API_URL}/DIMENSION/{dimension}/DimensionValues")
            return {item["Code"]: item["Title"] for item in values}
        except Exception as e:
            logger.error(
                f"WHO:: Unable to get the titles for dimension {dimension}: {str(e)}"
            )
            return {}
