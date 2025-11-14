import logging
import os
import time
from functools import wraps
from logging.handlers import RotatingFileHandler
from typing import Any, Callable

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


@staticmethod
def configure_logger() -> None:
    """Configure the logger based on the environment.

    Calling instructions: import and run configure_logger() once at the start of the application.
    """
    log_file_path = "./logging/backend.log"
    max_log_size_bytes = 1 * 1024 * 1024 * 1024  # 1GB in bytes
    backup_count = 5  # Number of backup log files to keep
    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=max_log_size_bytes,
        backupCount=backup_count,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(module)s:%(lineno)d]: %(message)s"
        )
    )

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(
        logging.DEBUG
    )  # Set the desired log level for the root logger (DEBUG, INFO, etc.)

    # Add the RotatingFileHandler to the root logger
    root_logger.addHandler(file_handler)

    logging.info("Logging configured correctly")


@staticmethod
def profile_method(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to profile method execution time.

    Args:
        func (Callable[..., Any]): The method to be decorated.

    Returns:
        The input function if not in development, otherwise the wrapped function that logs execution time.
    """
    is_debug = os.getenv("ENV", "prod").lower() in ["dev", "development"]
    if not is_debug:
        return func

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"⏱️  {func.__module__}:{func.__name__} took {elapsed:.4f}s")
        return result

    return wrapper


@staticmethod
def json_return(
    code, data: Any = None, error: Any = None, error_message: str = None
) -> tuple[dict, int]:
    """Return a jsonify-d flask object with the provided error code. Defaults to 204.

    Args:
        param code: the HTTP status code
    :param data: the data to return, can be an object, or a string
    """
    if error:
        return {"code": code, "error": error, "error_message": error_message}, code
    if data or data == []:
        return {"code": code, "result": data}, code
    else:
        return {
            "code": 204,
            "result": "We were unable to process, please contact your system administrator.",
        }, 204


@staticmethod
@profile_method
def setup_parsed_loc(ds_loc: str) -> str:
    """Set up the location for parsed datasets.

    Returns:
        str: The path to the parsed datasets.
    """
    logger.debug("Setting up parsed location")
    try:
        return ds_loc
    except Exception:
        return "./"


DATE_FORMATS = [
    "%d %b %Y",  # d MMM yyyy
    "%d %b %Y %H:%M:%S",  # d MMM yyyy HH:mm:ss
    "%d %b %Y %H:%M:%S %z",  # d MMM yyyy HH:mm:ss Z
    "%d %B %Y",  # d MMMM yyyy
    "%d %B %Y %H:%M:%S %Z",  # d MMMM yyyy HH:mm:ss z
    "%d-%m-%y",  # d-M-yy
    "%d-%m-%y %H:%M",  # d-M-yy H:mm
    "%d.%m.%y %H:%M",  # d.M.yy H:mm
    "%d/%m/%Y",  # d/M/yyyy
    "%d/%m/%Y %I:%M %p",  # d/M/yyyy h:mm a
    "%d/%m/%Y %I:%M:%S %p",  # d/M/yyyy h:mm:ss a
    "%d/%m/%Y %H:%M",  # d/M/yyyy HH:mm
    "%d/%m/%Y %H:%M:%S",  # d/M/yyyy HH:mm:ss
    "%d/%m/%y",  # d/MM/yy
    "%d/%m/%y %H:%M",  # d/MM/yy H:mm
    "%d/%b/%Y %H:%M:%S %z",  # d/MMM/yyyy H:mm:ss Z
    "%d %B %Y",  # dd MMMM yyyy
    "%d %B %Y %H:%M:%S %Z",  # dd MMMM yyyy HH:mm:ss z
    "%d-%b-%y %I.%M.%S.%f %p",  # dd-MMM-yy hh.mm.ss.nnnnnnnnn a
    "%d-%b-%Y",  # dd-MMM-yyyy
    "%d-%b-%Y %H:%M:%S",  # dd-MMM-yyyy HH:mm:ss
    "%d/%m/%y",  # dd/MM/yy
    "%d/%m/%y %H:%M",  # dd/MM/yy HH:mm
    "%d/%m/%y %H:%M:%S",  # dd/MM/yy HH:mm:ss
    "%d/%m/%Y",  # dd/MM/yyyy
    "%d/%m/%Y %I:%M %p",  # dd/MM/yyyy h:mm a
    "%d/%m/%Y %I:%M:%S %p",  # dd/MM/yyyy h:mm:ss a
    "%d/%m/%Y %H:%M",  # dd/MM/yyyy HH:mm
    "%d/%m/%Y %H:%M:%S",  # dd/MM/yyyy HH:mm:ss
    "%d/%b/%y %I:%M %p",  # dd/MMM/yy h:mm a
    "%a %b %d %H:%M:%S %Z %Y",  # EEE MMM dd HH:mm:ss z yyyy
    "%a, %d %b %Y %H:%M:%S %z",  # EEE, d MMM yyyy HH:mm:ss Z
    "%A %d %B %Y",  # EEEE d MMMM yyyy
    "%A %d %B %Y %H h %M %Z",  # EEEE d MMMM yyyy H' h 'mm z
    "%A %d %B %Y %H h %M %Z",  # EEEE d MMMM yyyy HH' h 'mm z
    "%A, %d %B %Y",  # EEEE, d MMMM yyyy
    # EEEE, d MMMM yyyy HH:mm:ss 'o''clock' z
    "%A, %d %B %Y %H:%M:%S o'clock %Z",
    "%A, %B %d, %Y",  # EEEE, MMMM d, yyyy
    "%A, %B %d, %Y %I:%M:%S %p %Z",  # EEEE, MMMM d, yyyy h:mm:ss a z
    "%m-%d-%y",  # M-d-yy
    "%m-%d-%y %I:%M %p",  # M-d-yy h:mm a
    "%m-%d-%y %I:%M:%S %p",  # M-d-yy h:mm:ss a
    "%m-%d-%y %H:%M",  # M-d-yy HH:mm
    "%m-%d-%y %H:%M:%S",  # M-d-yy HH:mm:ss
    "%m-%d-%Y",  # M-d-yyyy
    "%m-%d-%Y %I:%M %p",  # M-d-yyyy h:mm a
    "%m-%d-%Y %I:%M:%S %p",  # M-d-yyyy h:mm:ss a
    "%m-%d-%Y %H:%M",  # M-d-yyyy HH:mm
    "%m-%d-%Y %H:%M:%S",  # M-d-yyyy HH:mm:ss
    "%m/%d/%y",  # M/d/yy
    "%m/%d/%y %I:%M %p",  # M/d/yy h:mm a
    "%m/%d/%y %H:%M:%S",  # M/d/yy H:mm:ss (HH:mm:ss, MM)
    "%m/%d/%y %H:%M",  # M/d/yy HH:mm
    "%m/%d/%Y",  # M/d/yyyy
    "%m/%d/%Y %I:%M %p",  # M/d/yyyy h:mm a
    "%m/%d/%Y %I:%M:%S %p",  # M/d/yyyy h:mm:ss a
    "%m/%d/%Y %H:%M",  # M/d/yyyy HH:mm
    "%m/%d/%Y %H:%M:%S",  # M/d/yyyy HH:mm:ss
    "%m-%d-%y",  # MM-dd-yy
    "%m-%d-%y %I:%M %p",  # MM-dd-yy h:mm a
    "%m-%d-%y %I:%M:%S %p",  # MM-dd-yy h:mm:ss a
    "%m-%d-%y %H:%M",  # MM-dd-yy HH:mm
    "%m-%d-%y %H:%M:%S",  # MM-dd-yy HH:mm:ss
    "%m-%d-%Y",  # MM-dd-yyyy
    "%m-%d-%Y %I:%M %p",  # MM-dd-yyyy h:mm a
    "%m-%d-%Y %I:%M:%S %p",  # MM-dd-yyyy h:mm:ss a
    "%m-%d-%Y %H:%M",  # MM-dd-yyyy HH:mm
    "%m-%d-%Y %H:%M:%S",  # MM-dd-yyyy HH:mm:ss
    "%m/%d/%y",  # MM/dd/yy
    "%m/%d/%y %I:%M %p",  # MM/dd/yy h:mm a
    "%m/%d/%y %I:%M:%S %p",  # MM/dd/yy h:mm:ss a
    "%m/%d/%y %H:%M",  # MM/dd/yy HH:mm
    "%m/%d/%Y",  # MM/dd/yyyy
    "%m/%d/%Y %I:%M %p",  # MM/dd/yyyy h:mm a
    "%m/%d/%Y %I:%M:%S %p",  # MM/dd/yyyy h:mm:ss a
    "%m/%d/%Y %H:%M",  # MM/dd/yyyy HH:mm
    "%m/%d/%Y %H:%M:%S",  # MM/dd/yyyy HH:mm:ss
    "%b %d %Y",  # MMM d yyyy
    "%b %d, %Y",  # MMM d, yyyy
    "%b %d, %Y %I:%M:%S %p",  # MMM d, yyyy h:mm:ss a
    "%b.%d.%Y",  # MMM.dd.yyyy
    "%B %d %Y",  # MMMM d yyyy
    "%B %d, %Y",  # MMMM d, yyyy
    "%B %d, %Y %I:%M:%S %p %Z",  # MMMM d, yyyy h:mm:ss z a
    "%y-%m-%d",  # yy-MM-dd
    "%Y-'W'%W-%w",  # YYYY-'W'w-c
    "%Y-%j%z",  # yyyy-DDDXXX
    "%Y-%m-%d %I:%M %p",  # yyyy-M-d h:mm a
    "%Y-%m-%d %I:%M:%S %p",  # yyyy-M-d h:mm:ss a
    "%Y-%m-%d %H:%M",  # yyyy-M-d HH:mm
    "%Y-%m-%d %H:%M:%S",  # yyyy-M-d HH:mm:ss
    "%Y-%m-%d",  # yyyy-MM-dd
    "%Y-%m-%d %G",  # yyyy-MM-dd G
    "%Y-%m-%d %I:%M %p",  # yyyy-MM-dd h:mm a
    "%Y-%m-%d %I:%M:%S %p",  # yyyy-MM-dd h:mm:ss a
    "%Y-%m-%d %H:%M:%S",  # yyyy-MM-dd HH:mm:ss
    "%Y-%m-%d %H:%M:%S,%f",  # yyyy-MM-dd HH:mm:ss,SSS
    "%Y-%m-%d %H:%M:%S,%f[%Z]",  # yyyy-MM-dd HH:mm:ss,SSS'['VV']'
    "%Y-%m-%d %H:%M:%S,%fZ",  # yyyy-MM-dd HH:mm:ss,SSS'Z'
    "%Y-%m-%d %H:%M:%S,%f%z",  # yyyy-MM-dd HH:mm:ss,SSSXXX
    "%Y-%m-%d %H:%M:%S,%f%z[%Z]",  # yyyy-MM-dd HH:mm:ss,SSSXXX'['VV']'
    "%Y-%m-%d %H:%M:%S.%f",  # yyyy-MM-dd HH:mm:ss.S
    "%Y-%m-%d %H:%M:%S.%f",  # yyyy-MM-dd HH:mm:ss.SSS
    "%Y-%m-%d %H:%M:%S.%f[%Z]",  # yyyy-MM-dd HH:mm:ss.SSS'['VV']'
    "%Y-%m-%d %H:%M:%S.%fZ",  # yyyy-MM-dd HH:mm:ss.SSS'Z'
    "%Y-%m-%d %H:%M:%S.%f%z",  # yyyy-MM-dd HH:mm:ss.SSSXXX
    "%Y-%m-%d %H:%M:%S.%f%z[%Z]",  # yyyy-MM-dd HH:mm:ss.SSSXXX'['VV']'
    "%Y-%m-%d %H:%M:%S%z[%Z]",  # yyyy-MM-dd HH:mm:ssXXX'['VV']'
    "%Y-%m-%d %H:%M:%S%z",  # yyyy-MM-dd HH:mm:ssZ (ss'Z', ssX, ssXXX)
    "%Y-%m-%dT%H:%M:%S",  # yyyy-MM-dd'T'HH:mm:ss
    "%Y-%m-%dT%H:%M:%S,%f",  # yyyy-MM-dd'T'HH:mm:ss,SSS
    "%Y-%m-%dT%H:%M:%S,%f[%Z]",  # yyyy-MM-dd'T'HH:mm:ss,SSS'['VV']'
    "%Y-%m-%dT%H:%M:%S,%fZ",  # yyyy-MM-dd'T'HH:mm:ss,SSS'Z'
    "%Y-%m-%dT%H:%M:%S,%f%z",  # yyyy-MM-dd'T'HH:mm:ss,SSSXXX
    "%Y-%m-%dT%H:%M:%S,%f%z[%Z]",  # yyyy-MM-dd'T'HH:mm:ss,SSSXXX'['VV']'
    "%Y-%m-%dT%H:%M:%S.%f",  # yyyy-MM-dd'T'HH:mm:ss.SSS
    "%Y-%m-%dT%H:%M:%S.%f[%Z]",  # yyyy-MM-dd'T'HH:mm:ss.SSS'['VV']'
    "%Y-%m-%dT%H:%M:%S.%fZ",  # yyyy-MM-dd'T'HH:mm:ss.SSS'Z'
    "%Y-%m-%dT%H:%M:%S.%f%z",  # yyyy-MM-dd'T'HH:mm:ss.SSSXXX
    "%Y-%m-%dT%H:%M:%S.%f%z[%Z]",  # yyyy-MM-dd'T'HH:mm:ss.SSSXXX'['VV']'
    "%Y-%m-%dT%H:%M:%S%z[%Z]",  # yyyy-MM-dd'T'HH:mm:ssXXX'['VV']'
    "%Y-%m-%dT%H:%M:%S%z",  # yyyy-MM-dd'T'HH:mm:ssZ (ss'Z', ssX, ssXXX)
    "%Y-%m-%d%z",  # yyyy-MM-ddXXX
    "%Y'W'%W%w",  # YYYY'W'wc
    "%Y/%m/%d",  # yyyy/M/d
    "%Y%m%d%z",  # yyyyMMddZ
    "%Y-W%V",  # Week: 2024-W34
    "%G-W%V-%u",  # Week with weekday: 2024-W34-5
    "%Y-%j",  # Ordinal date: 2024-236
]
