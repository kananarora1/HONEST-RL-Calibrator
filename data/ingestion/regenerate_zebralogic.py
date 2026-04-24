"""Regenerate ZebraLogic-style constraint-satisfaction puzzles from seeds.

Produces fresh logic puzzles by sampling attribute/house grids and solving
them with a CSP backend so each problem ships with a verified unique
solution. Outputs unified JSONL records into ``data/processed/logic/``.
"""
