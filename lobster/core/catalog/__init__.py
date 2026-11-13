"""Optional workspace catalog.

Lobster records saved datasets, sessions and exported notebooks in an external
catalog through the small port in ``port.py``. Core code only ever sees that port:
``NullCatalog`` (the default) does nothing, and an engine runs behind
``ThreadedCatalog`` on one worker thread, failing open. ``factory.get_catalog`` decides
which one a process gets.
"""
