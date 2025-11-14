import concurrent.futures
import logging
from abc import ABC

from rb_core_backend.external_sources.util import LEGACY_EXTERNAL_DATASET_FORMAT
from rb_core_backend.mongo import RBCoreBackendMongo
from rb_core_backend.util import profile_method

logger = logging.getLogger(__name__)
DEFAULT_SEARCH_TERM = "World population"
DATA_FILE = " - Data file: "


class RBCoreExternalSources(ABC):
    """RB Core External Sources class for operations."""

    @profile_method
    def __init__(self, mongo_client: RBCoreBackendMongo, all_sources: dict) -> None:
        """Initialize the RBCoreExternalSources class with a mongo client, and user defined sources.
        Args:
            mongo_client (RBCoreBackendMongo): The MongoDB client instance.
            all_sources (dict, optional): A dictionary of all sources to index.
        """
        self.mongo_client = mongo_client
        self.all_sources = all_sources

    @profile_method
    def external_search_index(self) -> str:
        """Trigger the individual index functions for each source.
        Once they are all done, create a text index for the external sources.

        Returns:
            str: A string indicating the result of the indexing.
        """
        logger.info("Indexing external sources...")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Fire the index functions at the same time

            # Example implementation per source
            futures = {
                source: executor.submit(
                    actions["index"], mongo_client=self.mongo_client, delete=True
                )
                for source, actions in self.all_sources.items()
            }

            # Wait for all the index functions to finish
            concurrent.futures.wait(futures.values())

            # Log results for debugging
            for source, future in futures.items():
                try:
                    result = future.result()
                    logger.info(f"{source} index result: {result}")
                except Exception as e:
                    logger.error(f"{source} index failed: {e}")

        success = self.mongo_client.mongo_create_text_index_for_external_sources()
        logger.info("Done indexing external sources...")
        if success:
            return "Indexing successful"
        else:
            return "Indexing failed"

    @profile_method
    def external_search_force_reindex(self, source: str) -> str:
        """Shorthand function to force reindex a source.

        Args:
            source (str): The source to reindex.

        Returns:
            str: A string indicating the result of the reindexing.
        """
        self.all_sources[source]["index"](self.mongo_client, delete=True)

        success = self.mongo_client.mongo_create_text_index_for_external_sources()
        if success:
            return "Indexing successful"
        else:
            return "Indexing failed"

    @profile_method
    def search_external_sources(
        self,
        query: str,
        sources: list[str],
        legacy: bool = False,
        limit: int = None,
        offset: int = 0,
        sort_by: str = None,
    ) -> list[dict] | str:
        """Given a query, find all results in mongoDB from FederatedSearchIndex.

        Args:
            query (str): The query to search for.
            sources (list[str], optional): A list of sources to search through.
            legacy (bool, optional): A indicating if the search is to be returned legacy format. Defaults to False.
            limit (int, optional): The maximum number of results to return. Defaults to None.
            offset (int, optional): The offset to start the search from. Defaults to 0.
            sort_by (str, optional): The field to sort by. Defaults to None.

        Returns:
            list[dict] | str: A list of results in the form of an ExternalSource object or a string with an error.
        """
        try:
            if not sources:  # if sources is empty, set to all sources
                sources = list(self.all_sources.keys())
            if query == "":
                res = self.mongo_client.mongo_get_external_source_by_source(
                    sources, limit=limit, offset=offset, sort_by=sort_by
                )
            else:
                res = self.mongo_client.mongo_find_external_sources_by_text(
                    query, limit=limit, offset=offset, sources=sources, sort_by=sort_by
                )
            # Remove the 'score' and '_id' from every item in res and filter by source
            res = [
                {k: v for k, v in item.items() if k not in ("score", "_id")}
                for item in res
                if item.get("source") in sources
            ]
            res = [item for item in res if self._validate_search_result(item, query)]
            # For legacy requests, convert the results
            if legacy:
                res = self._convert_legacy_search_results(res)
        except Exception as e:
            logger.error(f"Error in external source search: {str(e)}")
            res = "Sorry, we were unable to search the external sources, please try again with a different search term, or contact the admin for more information."  # noqa
        return res

    @profile_method
    def download_external_source(self, external_dataset: dict) -> str:
        """
        This process should receive an external dataset object, and download
        to a staging folder. Then process the dataset. If anything fails, the
        dataset should be removed from the staging folder. This dataset is then
        included in the DX Mongo datasets. Then, the dataset is to be processed as a standard DX dataset.

        Args:
            external_dataset (dict): An external dataset object

        Returns:
            str: A string indicating the result of the download
        """
        result = "Sorry, we could not find the data source, please contact the admin for more information."
        try:
            logger.info(
                f"Downloading external dataset {external_dataset['name']} from {external_dataset['source']}."
            )
            source = external_dataset["source"]
            if source in self.all_sources:
                result = self.all_sources[source]["download"](external_dataset)
        except Exception as e:
            logger.error(f"Error in external source download: {str(e)}")
            result = "Sorry, we were unable to download your selected file. Contact the admin for more information."
        return result

    @staticmethod
    @profile_method
    def _validate_search_result(item: dict, query: str) -> bool:
        """Validate the search result against the query.
        This function checks if the query is present in the title or description of the item.

        Args:
            item (dict): The search result item to validate.
            query (str): The query text to check against.

        Returns:
            bool: True if the item is valid, False otherwise.
        """
        # If the query is empty, or contains quotes, we return True to include all results.
        if not query or '"' in query:
            return True
        query = query.lower()
        # split query on spaces and commas
        query = query.split()
        query = [q.strip() for q in query if q.strip()]  # remove empty strings
        logger.info("validating result with query: %s", query)
        title = item.get("title", "").lower()
        description = item.get("description", "").lower()
        logger.info("title: %s, description: %s", title, description)
        for q in query:
            if q not in title and q not in description:
                print("Q not in title or description:", q, title, description)
                return False
        return True

    @staticmethod
    @profile_method
    def _convert_legacy_search_results(all_res) -> list[dict]:
        """Convert to the expected format for frontend, and return.

        Conversions required:
            title to name
            URI to url
            mainCategory to category

            Create a duplicate for each resource, and combine the names.

        Args:
            all_res (list[dict]): List of all results from the search.

        Returns:
            list[dict]: A converted list of search results.
        """
        all_new_res = []
        for res in all_res:
            one_file = res.get("resources", []) == 1
            for resource in res.get("resources", []):
                new_res = LEGACY_EXTERNAL_DATASET_FORMAT.copy()
                new_res["name"] = res.get("title")
                new_res["description"] = res.get("description")
                if not one_file:
                    new_res["name"] += DATA_FILE + resource.get("title")
                    new_res["description"] += DATA_FILE + resource.get("title")
                if "QuickCharts" in new_res["name"]:
                    continue
                new_res["source"] = res.get("source")
                new_res["url"] = res.get("URI")
                new_res["category"] = res.get("mainCategory")
                new_res["datePublished"] = res.get("datePublished")
                new_res["owner"] = "externalSource"
                new_res["authId"] = "externalSource"
                new_res["public"] = False
                all_new_res.append(new_res)
        return all_new_res
