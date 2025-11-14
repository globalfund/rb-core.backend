# Logging

Calling instructions: import and run configure_logger() once at the start of the application.

```python
from util.configure_logging import configure_logger

configure_logger()
```

After running the code once, in `./logging/`, you should see a file `backend.log`, containing the line: `[INFO] [util:N]: Logging configured correctly`.
