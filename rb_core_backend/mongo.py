import logging
from abc import ABC

import pymongo

from rb_core_backend.util import profile_method

logger = logging.getLogger(__name__)


class RBCoreBackendMongo(ABC):
    """RB Core Backend MongoDB class for operations."""

    def __init__(
        self,
        mongo_host: str,
        mongo_username: str,
        mongo_password: str,
        mongo_auth_source: str,
        database_name: str,
        fs_db_name: str,
    ):
        logger.info("Initialized RBCoreBackendMongo")
        self.mongo_host = mongo_host
        self.mongo_username = mongo_username
        self.mongo_password = mongo_password
        self.mongo_auth_source = mongo_auth_source
        self.database_name = database_name
        self.fs_db_name = fs_db_name
        pass

    def mongo_client(self, dev: bool = False) -> pymongo.MongoClient:
        """Create a MongoDB client.

        Args:
            dev (bool, optional): Mode to use localhost:27017. Defaults to False.

        Returns:
            pymongo.MongoClient: A MongoDB client instance.
        """
        logger.info("Creating MongoDB client")
        if dev:
            return pymongo.MongoClient("localhost", 27017)
        else:
            return pymongo.MongoClient(
                self.mongo_host,
                username=self.mongo_username,
                password=self.mongo_password,
                authSource=self.mongo_auth_source,
            )

    @profile_method
    def mongo_create_external_source(
        self, external_source: dict, update: bool = False
    ) -> str | None:
        """Connect to the MongoDB and insert an external source object.

        Args:
            external_source (dict): The external source object to insert.
            update (bool, optional): Indicating if the object should be updated instead of inserted. Defaults to False.

        Returns:
            str | None: The inserted object ID or None if the operation failed.
        """
        logger.info("Creating or updating external source in MongoDB")
        try:
            client = self.mongo_client()
            db = client[self.database_name]
            external_source_collection = db[self.fs_db_name]
            if update:
                inserted_data = external_source_collection.update_one(
                    {"_id": external_source["_id"]}, {"$set": external_source}
                )
                client.close()
                return inserted_data.modified_count
            else:
                inserted_data = external_source_collection.insert_one(external_source)
                client.close()
                return inserted_data.inserted_id
        except Exception as e:
            logger.error("Error in create_external_source: " + str(e))
            return None

    @profile_method
    def mongo_create_external_sources(self, external_sources: list[dict]) -> int | None:
        """Connect to the MongoDB and insert multiple external source objects at once.

        Args:
            external_sources (list[dict]): The external source objects to insert.

        Returns:
            int | None: The number of inserted objects or None if the operation failed.
        """
        logger.info("Creating external sources in MongoDB")
        if not external_sources:
            return 0
        try:
            client = self.mongo_client()
            db = client[self.database_name]
            external_source_collection = db[self.fs_db_name]
            inserted_data = external_source_collection.insert_many(external_sources, ordered=False)
            client.close()
            return len(inserted_data.inserted_ids)
        except Exception as e:
            logger.error("Error in create_external_sources: " + str(e))
            return None

    @profile_method
    def mongo_count_external_sources(self) -> dict[str, int] | None:
        """Connect to the MongoDB and count the resources (data files) of the external sources per source.

        Returns:
            dict[str, int] | None: The number of resources per source, or None if the operation failed.
        """
        try:
            client = self.mongo_client()
            db = client[self.database_name]
            external_source_collection = db[self.fs_db_name]
            pipeline = [
                {"$group": {"_id": "$source", "count": {"$sum": {"$size": {"$ifNull": ["$resources", []]}}}}}
            ]
            counts = {item["_id"]: item["count"] for item in external_source_collection.aggregate(pipeline)}
            client.close()
            return counts
        except Exception as e:
            logger.error("Error in count_external_sources: " + str(e))
            return None

    @profile_method
    def mongo_get_all_external_sources(self) -> list[dict]:
        """Connect to the MongoDB and get all external source objects.

        Returns:
            list[dict]: A list of external source objects.
        """
        logger.info("Getting all external sources from MongoDB")
        try:
            client = self.mongo_client()
            db = client[self.database_name]
            external_source_collection = db[self.fs_db_name]
            external_sources = list(external_source_collection.find())
            client.close()
            return external_sources
        except Exception as e:
            logger.error("Error in get_all_external_sources: " + str(e))
            return []

    @profile_method
    def mongo_find_external_sources_by_text(
        self,
        query: str,
        limit: int = None,
        offset: int = 0,
        sources: list[str] = None,
        sort_by: str = None,
    ) -> list[dict]:
        """Connect to the MongoDB and find external source objects by title or description.

        Args:
            query (str): The query to search for.
            limit (int, optional) The maximum number of results to return.
            offset (int, optional): The offset to start the search from.
            sources (list[str], optional): A list of sources to filter by.
            sort_by (str, optional): The field to sort by. Defaults to None.

        Returns:
            list[dict]: A list of external source objects.
        """
        logger.info("Finding external sources by text in MongoDB")
        try:
            client = self.mongo_client()
            db = client[self.database_name]
            external_source_collection = db[self.fs_db_name]

            # Construct query to include text search and source filtering if sources are provided
            mongo_query = {"$text": {"$search": f"{query}"}}
            if sources:
                mongo_query["source"] = {"$in": sources}

            sort_style = [("score", {"$meta": "textScore"})]
            if sort_by == "updatedDate":
                sort_style = [("dateLastUpdated", -1)]
            if sort_by == "createdDate":
                sort_style = [("datePublished", -1)]
            if sort_by == "name":
                sort_style = [("title", 1)]

            external_sources = list(
                external_source_collection.find(
                    mongo_query, {"score": {"$meta": "textScore"}}
                )
                .sort(sort_style)
                .skip(offset or 0)
                # A limit of 0 is no limit in MongoDB, None is not accepted.
                .limit(limit or 0)
            )
            client.close()
            return external_sources
        except Exception as e:
            logger.error("Error in find_external_sources_by_text: " + str(e))
            return None

    @profile_method
    def mongo_get_external_source_by_source(
        self, sources, limit=None, offset=0, sort_by=None
    ) -> list[dict]:
        """Connect to the MongoDB and get external source objects by source.

        Args:
            sources (list[str]): A list of sources to filter by.
            limit (int, optional): The maximum number of results to return.
            offset (int, optional): The offset to start the search from.
            sort_by (str, optional): The field to sort by. Defaults to None.

        Returns:
            list[dict]: A list of external source objects.
        """
        try:
            client = self.mongo_client()
            db = client[self.database_name]
            external_source_collection = db[self.fs_db_name]

            sort_style = [("title", 1)]
            if sort_by == "name":
                sort_style = [("title", 1)]
            if sort_by == "updatedDate":
                sort_style = [("dateLastUpdated", -1)]
            if sort_by == "createdDate":
                sort_style = [("datePublished", -1)]

            external_sources = list(
                external_source_collection.find({"source": {"$in": sources}})
                # sort on alphabetical order
                .sort(sort_style)
                .skip(offset or 0)
                # A limit of 0 is no limit in MongoDB, None is not accepted.
                .limit(limit or 0)
            )
            client.close()
            return external_sources
        except Exception as e:
            logger.error("Error in get_external_source_by_source: " + str(e))
            return None

    @profile_method
    def mongo_create_text_index_for_external_sources(self) -> bool:
        """Connect to the MongoDB and create a text index for the external source objects.

        Returns:
            bool: A boolean indicating if the operation was successful.
        """
        try:
            client = self.mongo_client()
            db = client[self.database_name]
            external_source_collection = db[self.fs_db_name]
            external_source_collection.create_index(
                [
                    ("title", pymongo.TEXT),
                    ("description", pymongo.TEXT),
                    ("resources.title", pymongo.TEXT),
                    ("resources.description", pymongo.TEXT),
                ]
            )
            client.close()
            logger.info("Created text index for federated search results.")
            return True
        except Exception as e:
            logger.error("Error in create_text_index_for_external_sources: " + str(e))
            return False

    @profile_method
    def mongo_remove_data_for_external_sources(self, source=None) -> bool:
        """Connect to the MongoDB and remove data for the external source objects.
        Filtering on the provided source if it is provided. If it is not provided, drop all.

        Args:
            source (str, optional): The source to filter on. Defaults to None.

        Returns:
            bool: A boolean indicating if the operation was successful.
        """
        try:
            client = self.mongo_client()
            db = client[self.database_name]
            external_source_collection = db[self.fs_db_name]
            if source is None:
                external_source_collection.drop()
            else:
                external_source_collection.delete_many({"source": source})
            client.close()
            logger.info("Removed data for external sources.")
            return True
        except Exception as e:
            logger.error("Error in remove_data_for_external_sources: " + str(e))
            return False
