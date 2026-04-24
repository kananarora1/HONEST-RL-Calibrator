"""Verifier for logic puzzles.

Parses a candidate assignment and checks it against the puzzle's
constraints using python-constraint (or Z3 for harder instances),
confirming both constraint satisfaction and uniqueness where required.
"""
