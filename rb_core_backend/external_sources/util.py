# A global representation of the format of any external dataset.
# Definition of the external dataset
EXTERNAL_DATASET_FORMAT = {
    "title": "",  # External source dataset name
    "description": "",  # External source dataset description
    "source": "",  # External source name (Kaggle, etc)
    "URI": "",  # URL to read more about the dataset
    "internalRef": "",  # Internal reference to the dataset from the datasource
    "mainCategory": "",  # Category in which to place the dataset
    "subCategories": [],  # list of subcategories of the dataset
    "datePublished": "",  # Date the dataset was published
    "dateLastUpdated": "",  # Date this object was last updated
    "dateSourceLastUpdated": "",  # Date the dataset was last updated
    "resources": [],  # List of resources, in case there are multiple.
    "connectedDataset": [],  # Connected dataset objects, can be collections in the future
    "owner": "externalSource",
    "authId": "externalSource",
    "public": False,
}

# Definition of the external dataset resource
EXTERNAL_DATASET_RESOURCE_FORMAT = {
    "title": "",  # Resource name
    "description": "",  # Resource description
    "URI": "",  # URL to download the resource
    "internalRef": "",  # Internal reference to the resource from the datasource
    "format": "",  # Format of the resource
    "datePublished": "",  # Date the resource was published
    "dateLastUpdated": "",  # Date the resource was last updated
    "dateResourceLastUpdated": "",  # Date the resource was last updated in its source.
    "owner": "externalSource",
    "authId": "externalSource",
    "public": False,
}

# Legacy format for external datasets (deprecated)
LEGACY_EXTERNAL_DATASET_FORMAT = {
    "name": "",
    "description": "",
    "source": "",
    "url": "",
    "category": "",
    "datePublished": "",
    "owner": "externalSource",
    "authId": "externalSource",
    "public": False,
}
