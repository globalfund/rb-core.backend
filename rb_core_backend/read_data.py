"""
This file reads in a data file, not necessarily determined already.
It will be used to read in data from the staging directory, and then parse it into a dataframe.

First, we determine the file's encoding, which we use for reading the content.
Second, we determine the file's extension, which we use to determine how to parse the file.
Third, we parse the file into a dataframe, and return it.

For the several types of data files, we use a switch statement to select the correct parsing method.
"""

import datetime
import json
import logging
import mimetypes
import os
import sqlite3
import xml.etree.ElementTree as ET

import chardet
import magic
import pandas as pd
import requests
from sqlalchemy import create_engine

from rb_core_backend.util import profile_method

logger = logging.getLogger(__name__)


class RBCoreDataFileReader:
    """RB Core Data File Reader class for reading data files into pandas DataFrames."""

    def __init__(
        self,
        file_path: str,
        file_extension: str = "",
        file_encoding: str = None,
        optional_args: dict = {},
    ) -> None:
        """Initialize the RBCoreDataReader class.

        Args:
            file_path (str): The path to the datafile
            file_extension (str): The file's extension if provided
            file_encoding (str): The file's encoding if provided
            optional_args (dict): Optional supporting the data retrieval.
        """
        self.file_path = file_path
        self.file_extension = file_extension
        self.file_encoding = file_encoding
        self.optional_args = optional_args

    @profile_method
    def read_data(self) -> tuple[pd.DataFrame | str, str]:
        """Read data from a source into a pandas DataFrame.

        Returns:
            (pandas.DataFrame, str) | (str, str): A pandas dataframe with a message | "error" messages.
        """
        logger.info(f"Reading data from file: {self.file_path}")
        try:
            if self.file_path in ["MySQL", "PostgreSQL", "MsSQL", "Oracle"]:
                self.file_extension = self.file_path
                self.file_encoding = self.file_path
            else:
                self.file_extension = self._detect_file_extension()
                if self.file_extension == "error":
                    return "error", "error"

            try:
                res = self._read_path_as_df()
            except Exception:
                self.file_encoding = self._detect_file_encoding()
                res = self._read_path_as_df()
            if isinstance(res, tuple):
                df, message = res
            else:
                df = res
                message = "Successfully read dataframe."
            logger.debug(f"Dataset read finished with message: {message}")
            logger.debug(f"Dataframe:\n{df.head()}")
            return df, message
        except Exception as e:
            logger.error(f"Error reading data: {e}")
            return "error", "error"

    @profile_method
    def _detect_file_encoding(self) -> str:
        """Detect the encoding of a file

        Returns:
            str: The detected encoding of the file.
        """
        logger.debug(f"Detecting encoding of file: {self.file_path}")
        try:
            with open(self.file_path, "rb") as f:
                _start = datetime.datetime.now()
                f_read = f.read()
                print(
                    f"File read for encoding detection took: {datetime.datetime.now() - _start}"
                )
                _start = datetime.datetime.now()
                result = chardet.detect(f_read)
                print(f"chardet detection took: {datetime.datetime.now() - _start}")
            logger.debug(f"Detected encoding: {result['encoding']}")
            return result["encoding"]
        except Exception as e:
            logger.error(f"Error detecting file encoding: {e}")
            # Default to the most common file encoding
            return "utf-8"

    @profile_method
    def _detect_file_extension(self) -> str:
        """Detect the file type of a file. First, on any extension in the filename (used primarily for .h5 or .spss etc)
        if that does not work, use the mimetypes library, and if that does not work, use the magic library.

        Returns:
            str: The detected file extension of the file or error
        """
        try:
            if bool(os.path.splitext(self.file_path)[1]):
                t = os.path.splitext(self.file_path)[1]
            else:
                t = mimetypes.guess_type(self.file_path)[0]
                if t is None:
                    t = magic.from_file(self.file_path, mime=True)
                t = self._mimetype_to_extension(t)
            logger.debug(f"Detected file extension: {t}")
            return t
        except Exception as e:
            logger.error(f"Error detecting file extension: {e}")
            return "error"

    @staticmethod
    @profile_method
    def _mimetype_to_extension(mimetype) -> str:
        """Return the file extension for a given mimetype.

        Args:
            mimetype (string): the mimetype passed by _detect_file_extension.

        Returns:
            str: the mimetype or error
        """
        mimetype_extension_dict = {
            "text/csv": ".csv",
            "application/excel": ".xls",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
            "application/vnd.ms-excel.sheet.macroEnabled.12": ".xlsm",
            "application/vnd.ms-excel.sheet.binary.macroEnabled.12": ".xlsb",
            "application/vnd.oasis.opendocument.formula": ".odf",
            "application/vnd.oasis.opendocument.spreadsheet": ".ods",
            "application/vnd.oasis.opendocument.text": ".odt",
            "application/json": ".json",
            "application/xml": ".xml",
            "application/x-stata-data": ".dta",
            "application/x-hdf": ".h5",
        }
        try:
            return mimetype_extension_dict[mimetype]
        except KeyError:
            return "error"

    @profile_method
    def _read_path_as_df(self) -> pd.DataFrame | str | tuple[pd.DataFrame | str, str]:
        """Defines read functions for the supported input types, and reads the content accordingly.

        Raises:
            ValueError: if the file extension is not supported.

        Returns:
            pandas.DataFrame | str | tuple[pandas.DataFrame | str, str]:
                A pandas dataframe with the data or "error" in case of failure.
        """
        read_functions = {
            ".csv": lambda: pd.read_csv(
                self.file_path,
                encoding=self.file_encoding,
                low_memory=False,
                **self.optional_args,
            ),
            ".tsv": lambda: pd.read_csv(
                self.file_path,
                encoding=self.file_encoding,
                sep="\t",
                low_memory=False,
                **self.optional_args,
            ),
            ".json": lambda: self._read_json(),
            ".xml": lambda: self._read_xml(),
            ".xls": lambda: pd.read_excel(
                self.file_path, engine="xlrd", **self.optional_args
            ),
            ".xlsx": lambda: pd.read_excel(
                self.file_path, engine="openpyxl", **self.optional_args
            ),
            ".xlsm": lambda: pd.read_excel(
                self.file_path, engine="openpyxl", **self.optional_args
            ),
            ".xlsb": lambda: pd.read_excel(
                self.file_path, engine="pyxlsb", **self.optional_args
            ),
            ".ods": lambda: pd.read_excel(
                self.file_path, engine="odf", **self.optional_args
            ),  # excluding .odt and .odf for now
            ".dta": lambda: pd.read_stata(self.file_path, **self.optional_args),
            ".h5": lambda: pd.read_hdf(self.file_path, **self.optional_args),
            ".orc": lambda: pd.read_orc(
                self.file_path, **self.optional_args
            ),  # pyarrow
            ".feather": lambda: pd.read_feather(
                self.file_path, **self.optional_args
            ),  # pyarrow
            ".parquet": lambda: pd.read_parquet(
                self.file_path, **self.optional_args
            ),  # pyarrow
            ".sav": lambda: pd.read_spss(
                self.file_path, **self.optional_args
            ),  # pip install pyreadstat
            ".sas7bdat": lambda: pd.read_sas(self.file_path, **self.optional_args),
            ".sas": lambda: pd.read_sas(self.file_path, **self.optional_args),
            ".sqlite": lambda: self._read_sqlite(),
            "MySQL": lambda: self._read_sql(),  # pip install pymysql sqlalchemy
            "PostgreSQL": lambda: self._read_sql(),  # pip install psycopg sqlalchemy
            "MsSQL": lambda: self._read_sql(),  # pip install pymssql sqlalchemy
            "Oracle": lambda: self._read_sql(),  # pip install oracledb sqlalchemy
            # Introduce more from https://docs.sqlalchemy.org/en/13/dialects/index.html
        }

        logger.debug(
            f"Reading file {self.file_path} with extension {self.file_extension} and encoding {self.file_encoding}"
        )
        if self.file_extension in read_functions:
            try:
                res = read_functions[self.file_extension]()
                return res
            except Exception as e:
                logger.error(f"Error reading file: {e}")
                return "error"
        else:
            raise ValueError("Unsupported file extension")

    @profile_method
    def _read_json(self) -> pd.DataFrame:
        """Read a json file into a dataframe, but normalize it first.
        Achieved by json loading the json file, then applying pandas.json_normalize to it.

        There are some notes in ./read_data_notes.md about the json files.

        optional_args (dict): Optional arguments to pass to pandas.json_normalize:
            data: dict | list[dict],
            record_path: str | list | None = None,
            meta: str | list[str | list[str]] | None = None,
            meta_prefix: str | None = None,
            record_prefix: str | None = None,
            errors: IgnoreRaise = "raise",
            sep: str = ".",
            max_level: int | None = None,

        Returns:
            pandas.DataFrame: A pandas dataframe with the data.
        """
        with open(self.file_path, "r") as f:
            data = json.load(f)
        return pd.json_normalize(data, **self.optional_args)

    @profile_method
    def _read_xml(self) -> tuple[pd.DataFrame, str]:
        """
        Read a XML file into a dataframe, but normalize it first.
        Achieved by json loading the xml file, then applying pandas.json_normalize to it.

        There are some notes in ./read_data_notes.md about the xml files.

        optional_args (dict): Optional arguments to pass to pd.read_xml.
            namespaces: dict[str, str] | None = None,
            elems_only: bool = False,
            attrs_only: bool = False,
            names: Sequence[str] | None = None,
            dtype: DtypeArg | None = None,
            converters: ConvertersArg | None = None,
            parse_dates: ParseDatesArg | None = None,
            # encoding can not be None for lxml and StringIO input
            parser: XMLParsers = "lxml",
            stylesheet: FilePath | ReadBuffer[bytes] | ReadBuffer[str] | None = None,
            iterparse: dict[str, list[str]] | None = None,
            compression: CompressionOptions = "infer",
            storage_options: StorageOptions | None = None,
            dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,

        Returns:
            (pandas.DataFrame, str): A pandas dataframe with the data and a message about nested data.
        """
        message = "Successfully read XML file."
        if not self._is_shallow_xml():
            message = "Nested XML data is not supported. Please create a shallow XML file of the nested data if you wish to use it."  # noqa: E501
        return (
            pd.read_xml(self.file_path, encoding=self.encoding, **self.optional_args),
            message,
        )

    @profile_method
    def _is_shallow_xml(self) -> bool:
        """
        Function to determine the shallowness of an XML file, i.e. whether it contains nested data or not.
        The nested 'limit' is two levels, as this allows us to create a 2d table from the data.

        Returns:
            bool: True if the XML file is shallow, False otherwise.
        """
        try:
            with open(self.file_path, "r") as xml_file:
                xml_content = xml_file.read()
                root = ET.fromstring(xml_content)
                for element in root:
                    for item in element:
                        if len(item) > 0:
                            return False
                return True
        except ET.ParseError as e:
            logger.error(f"Error parsing XML: {e}")
            return False

    @profile_method
    def _read_sqlite(self) -> tuple[None, str] | pd.DataFrame:
        """Read a sqlite database file into a dataframe.

        Args:
            file_path (str): The path to the sqlite file.
            optional_args (dict): Optional arguments to pass to the sqlite3 connection.
                table: str - The table to read from the sqlite database.

        Returns:
            tuple[None, str] | pandas.DataFrame: A pandas dataframe with the data or None and an error message.
        """
        conn = sqlite3.connect(self.file_path)
        if "table" in self.optional_args:
            df = pd.read_sql_query(f"SELECT * FROM {self.optional_args['table']}", conn)
        else:
            return None, "No table specified"
        return df

    @profile_method
    def _read_sql(self) -> tuple[None | pd.DataFrame, str]:
        """Read a SQL database into a dataframe.

        optional_args (dict): Optional arguments to pass to the SQL connection.
            username: str - The username to connect to the database.
            password: str - The password to connect to the database.
            host: str - The host of the database.
            port: str - The port of the database.
            database: str - The database name.
            query: str - The SQL query to execute.
            table: str - The table to read from the database (if query is not provided).

        Returns:
            tuple[None | pandas.DataFrame, str]: A pandas dataframe with the data or None and an error message.
        """
        try:
            username = self.optional_args["username"]
            password = self.optional_args["password"]
            host = self.optional_args["host"]
            port = self.optional_args["port"]
            database = self.optional_args["database"]
        except KeyError:
            return None, "Missing SQL credentials"

        try:
            query = self.optional_args["query"]
        except KeyError:
            try:
                table = self.optional_args["table"]
                query = f"SELECT * FROM {table}"
            except KeyError:
                return None, "Missing SQL query or table"

        if self.file_path == "MySQL":
            engine_type = "mysql+pymysql"
        elif self.file_path == "PostgreSQL":
            engine_type = "postgresql+psycopg"
        elif self.file_path == "MsSQL":
            engine_type = "mssql+pymssql"
        elif self.file_path == "Oracle":
            engine_type = "oracle+oracledb"
        else:
            return None, "Invalid database type"
        engine = f"{engine_type}://{username}:{password}@{host}:{port}/{database}"
        df = pd.read_sql(query, con=create_engine(engine))
        return df, "Successfully read SQL table."


class RBCoreDataAPIReader:
    """RB Core Data API Reader class for reading data files into pandas DataFrames."""

    def __init__(self, url: str, additional_args: dict = {}) -> None:
        """Initialize the RBCoreDataReader class.

        Args:
            url (str): The URL to download the data from
            additional_args (dict, optional): Additional arguments relating to the API.
                json_root: str - The root of the json data to extract
                xml_root: str - The root of the xml data to extract
        """
        self.url = url
        self.json_root = additional_args.get("json_root", None)
        self.xml_root = additional_args.get("xml_root", None)

    @profile_method
    def read_data_from_api(self) -> tuple[pd.DataFrame | str, str]:
        """Download data from an API

        Returns:
            tuple[pandas.DataFrame | str, str]: A pandas DataFrame with the data or "error" in case of failure.
        """
        logger.debug(
            f"download_api_data:: Starting download of API data with: \
             \n\turl: {self.url}\n\tjson_root: {self.json_root}\n\txml_root: {self.xml_root}"
        )
        try:
            # check if json_root is provided
            if self.json_root:
                logger.debug(f"self.json_root provided: {self.json_root}")
                df = self._download_and_prep_json()
            elif self.xml_root:
                logger.debug(f"xml_root provided: {self.xml_root}")
                df = self._download_and_prep_xml()
            else:
                logger.debug("No json_root or xml_root provided, assuming csv data")
                df = self._download_and_prep_csv()
            return df, "success"
        except Exception as e:
            logger.error(f"Error in download_api_data: {str(e)}")
            return "error", "error"

    @profile_method
    def _download_and_prep_json(self) -> pd.DataFrame:
        """Download and prepare json data from an API

        Returns:
            pandas.DataFrame: A pandas DataFrame with the data.
        """
        # Download the data
        data = requests.get(self.url).json()
        data = self._recursive_dict_root(data, self.json_root)
        df = pd.DataFrame(data)
        logger.debug(f"\n{df.head()}")
        return df

    @profile_method
    def _download_and_prep_xml(self) -> pd.DataFrame:
        """Download and prepare xml data from an API

        Returns:
            pandas.DataFrame: A pandas DataFrame with the data.
        """
        # convert xml_root to an XPATH
        if self.xml_root == ".":
            df = pd.read_xml(self.url)
        else:
            xpath = "/" + self.xml_root.replace(".", "/")
            df = pd.read_xml(self.url, xpath=xpath)
        logger.debug(f"\n{df.head()}")
        return df

    @profile_method
    def _download_and_prep_csv(self) -> pd.DataFrame:
        """Download and prepare csv data from an API

        Returns:
            pandas.DataFrame: A pandas DataFrame with the data.
        """
        df = pd.read_csv(self.url)
        logger.debug(f"\n{df.head()}")
        return df

    @profile_method
    def _recursive_dict_root(self, data, roots) -> dict | list:
        """Recursively extract the root from a nested dictionary.

        Args:
            data (dict | list): The nested dictionary or list to extract the root from.
            roots (str): The root to extract, separated by dots.

        Returns:
            dict | list: The extracted root.
        """
        if roots == "." or roots == "":
            return data
        else:
            root = roots.split(".")[0]
            return self._recursive_dict_root(data[root], ".".join(roots.split(".")[1:]))
