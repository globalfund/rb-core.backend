from abc import ABC, abstractmethod

from rb_core_backend.mongo import RBCoreBackendMongo
from rb_core_backend.preprocess_dataset import RBCoreDatasetPreprocessor


class ExternalSourceModel(ABC):
    """External Source Model class for operations."""

    def __init__(
        self,
        mongo_client: RBCoreBackendMongo,
        dataset_preprocessor: RBCoreDatasetPreprocessor,
    ) -> None:
        """Initialize the RBCoreExternalSourceKaggle class.

        Args:
            mongo_client (RBCoreBackendMongo): The MongoDB client to use for indexing.
            dataset_preprocessor (RBCoreDatasetPreprocessor): The dataset preprocessor to use for processing downloaded datasets.
        """  # NOQA: E501
        self.mongo_client = mongo_client
        self.dataset_preprocessor = dataset_preprocessor

    @abstractmethod
    def index(self, delete: bool = False) -> str:
        """Index the external source.

        Args:
            delete (bool, optional): Whether to delete existing entries before indexing. Defaults to False.

        Returns:
            str: A string indicating the result of the indexing.
        """
        pass

    @abstractmethod
    def download(self, external_dataset: dict) -> str:
        """Download a Kaggle dataset given an external dataset object. Using the Kaggle CLI with the dataset ref.

        Args:
            external_dataset (dict): An external dataset object

        Returns:
            str: A string indicating the result of the download (success, or error message)
        """
        pass
