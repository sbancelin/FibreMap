"""Single source of truth for code and data-schema versions.

``__version__`` tracks the codebase (semver). ``SCHEMA_VERSION`` tracks the on-disk
*bundle*/phantom schema (semver) and is written into every artifact for provenance and
migration. They evolve independently: a code change need not bump the schema, and a schema
change always bumps ``SCHEMA_VERSION``.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Bumped only when the Phantom / ImageBundle / bundle on-disk schema changes incompatibly.
SCHEMA_VERSION = "0.1.0"