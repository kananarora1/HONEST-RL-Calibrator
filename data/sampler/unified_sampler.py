"""Unified sampler exposing ingested problems behind the generator interface.

Loads JSONL shards from ``data/processed/`` across math/code/logic,
supports difficulty- and domain-weighted sampling, and yields problems
with the same shape the environment currently receives from the
procedural generators in ``server/generators/``, so it can be dropped in
without changes to the environment code.
"""
