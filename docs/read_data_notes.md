# Read Data Notes

## JSON Notes

Unfortunately, nested arrays are not supported, here is why:
Flattening nested arrays in a JSON file into a 2D table, such as a DataFrame, can also pose similar challenges and risks as flattening nested XML structures. Here's an explanation of why flattening nested arrays in JSON might be considered dangerous:

- Loss of Hierarchy: JSON is designed to represent structured and nested data using arrays and objects. Flattening nested arrays into a table can lead to the loss of the original hierarchical structure. Elements that were intended to be nested might lose their meaningful relationships and context.
- Data Duplication: Flattening can cause data duplication as values from higher levels of the hierarchy are repeated across multiple rows. This duplication can inflate the dataset's size and potentially lead to inconsistencies.
- Misinterpretation of Relationships: JSON arrays can represent lists of related data. Flattening can break these relationships, making it challenging to understand which elements were originally related.
- Loss of Metadata: Just like XML, JSON objects can contain metadata and attributes associated with their elements. Flattening might not properly capture this metadata, leading to information loss.
- Data Integrity Issues: When nested arrays are flattened, it might become difficult to maintain data integrity, especially during updates or modifications. Original relationships could be challenging to reconstruct.
- Loss of Context: The original context and structure of the data can be lost, making it harder to understand the meaning behind the data points.
- Complex Nested Structures: JSON arrays can represent complex hierarchical structures. Flattening these structures can result in overly wide tables, making the data difficult to work with and analyze effectively.

In cases where the nested arrays in a JSON file hold meaningful relationships and context, it's generally recommended to work with the JSON data while preserving its hierarchical structure. If you do need to work with tabular data, carefully consider whether flattening is appropriate for your specific use case and whether any crucial information will be compromised in the process. Using tools that can handle hierarchical JSON data can provide better insights and more accurate representations of the original data structure.

Your file may still get processed by the data parser, but your arrays will likely be strings with the array's literal contents.

## XML Notes

Unfortunately, nested XML data is not supported. Here is why:
Flattening nested XML structures into a 2D table (like a DataFrame) can lead to data loss and misinterpretation, especially when the XML structure is hierarchical and contains complex relationships between elements. XML is designed to represent hierarchical data, and attempting to flatten it into a tabular structure might not capture the full context and relationships present in the original XML.
When you flatten nested XML into a 2D table, you might encounter the following challenges:

- Data Loss: Flattening nested XML can lead to the loss of hierarchical relationships, resulting in a loss of context and meaning. Nested data structures might contain information that is crucial for understanding the data.
- Duplicated Data: Flattening can lead to duplication of data, as elements from higher levels of the hierarchy are replicated across multiple rows, potentially leading to redundancy and increased storage requirements.
- Misrepresentation: Complex relationships between elements might not be accurately represented in a tabular format, leading to misinterpretation of the data.
- Loss of Metadata: XML often includes attributes and metadata associated with elements, which might not be properly represented in a flattened table.
- Data Integrity: Maintaining data integrity, especially when performing updates or modifications, can become challenging when the original hierarchical structure is lost.
- Loss of Context: Hierarchical relationships can provide important context to the data. Flattening can make it harder to understand the original structure and relationships.

In cases where the XML structure is inherently hierarchical and contains nested elements with meaningful relationships, it's generally advisable to work with the XML data as is, using tools that allow you to navigate and query the hierarchical structure. If you do need to work with the data in a tabular format, consider carefully whether flattening is appropriate for your specific use case and whether any information or context will be lost in the process.

## API Reading

Description for the user:

Please provide us with the URL which provides access to the data in your API.
This can be in JSON, XML or CSV format.
We assume you have provided us with CSV if you do not specify.
We do not store the URL anywhere in our system, so any authentication credentials you may pass, will not be stored by us.

If you provide us with an URL that provides the data in CSV, we assume the data is to be used as-is.

If you chose to provide us with JSON or XML, please provide us with a "base field", which is the field that contains your data.
For example if the flat data is in value or result.docs.
If the data is in the root of the JSON or XML, please leave this field with exclusively a period(.) as specifier.

Note with JSON and XML that the following still applies:
Flattening nested XML structures into a 2D table (like a DataFrame) can lead to data loss and misinterpretation, especially when the XML structure is hierarchical and contains complex relationships between elements. XML is designed to represent hierarchical data, and attempting to flatten it into a tabular structure might not capture the full context and relationships present in the original XML.
Flattening nested arrays in a JSON file into a 2D table, such as a DataFrame, can also pose similar challenges and risks as flattening nested XML structures.
