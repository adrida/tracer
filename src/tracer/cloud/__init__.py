"""Tracer Cloud client + CLI command group.

`tracer cloud <subcommand>` mirrors the Tracer Cloud web UI from the
terminal: log in once, then create/train/retrain/promote tracers, run live
routing queries, manage test batteries, models, ingest keys, billing and
analytics. Authentication is a Tracer Cloud session (the same identity the
browser uses), presented to the web API as an `Authorization: Bearer <jwt>`
header.
"""

from .client import CloudClient, CloudError  # noqa: F401
