#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Reads a constant out of an analysis script without running it, so that a figure shows the value the analysis used rather than a
typed copy of it. figures_3_and_4.py takes the labels' departure distance and the window lengths from the scripts themselves.

  module_const(path, name)          the value of a module-level constant (read from the syntax tree; the file is not run)
  source_regex(path, pattern)       the first group of a regular expression in a source file (a literal inside a code line)
  function_literal(path, f, v, i)   the i-th numeric literal assigned to v inside function f
Usage   imported by codes/figures/figures_3_and_4.py
"""

from __future__ import annotations  # future import: type hints kept as text, so they cost nothing at run time

import ast  # standard library: reads constants out of the analysis scripts without running them
import math  # standard library: log10 for the decibel ratio
from pathlib import Path  # Path objects for explicit file paths

import pandas as pd  # pandas: reads the CSV data products

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/




OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b}  # dict: the arithmetic operators a constant may use -> how to apply them


def safe_value(node):  # function: evaluates a constant expression (numbers, + - * / **, dicts, tuples, lists) from the syntax tree, nothing else
    if isinstance(node, ast.Constant):  # if: a literal number or string
        return node.value  # its value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):  # if: a negative number
        return -safe_value(node.operand)  # recursion on the operand, negated
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:  # if: arithmetic between two constant expressions, such as 2 * 3600.0
        return OPS[type(node.op)](safe_value(node.left), safe_value(node.right))  # recursion on both sides, then the operator
    if isinstance(node, ast.Dict):  # if: a dictionary literal
        return {safe_value(k): safe_value(v) for k, v in zip(node.keys, node.values)}  # dict comprehension over the key and value nodes
    if isinstance(node, (ast.Tuple, ast.List)):  # if: a tuple or list literal
        return [safe_value(e) for e in node.elts]  # list comprehension over the elements
    raise ValueError(f"not a constant expression: {ast.dump(node)[:80]}")  # stops on anything else (a name, a call)


def module_const(path: str, name: str):  # function: the value of a module-level constant in a Python file, read with ast (the file is not run)
    tree = ast.parse((PROJECT / path).read_text(encoding="utf-8"))  # the file's syntax tree
    for node in tree.body:  # for loop over the top-level statements
        targets = [node.target] if isinstance(node, ast.AnnAssign) else getattr(node, "targets", [])  # conditional expression: annotated or plain assignment targets
        for t in targets:  # for loop over the names assigned
            if isinstance(t, ast.Name) and t.id == name:  # if: the constant looked for
                return safe_value(node.value)  # evaluates the constant expression without running the file
            if isinstance(t, ast.Tuple) and name in [getattr(e, "id", None) for e in t.elts]:  # if: a tuple assignment such as A, B = 30.0, 30
                return safe_value(node.value)[[e.id for e in t.elts].index(name)]  # the value at the same position on the right-hand side
    raise KeyError(f"{name} not found in {path}")  # stops when the constant is missing


def source_regex(path: str, pattern: str) -> str:  # function: the first group of the first match of a regular expression in a source file (for literals inside code lines)
    import re  # standard library: regular expressions
    m = re.search(pattern, (PROJECT / path).read_text(encoding="utf-8"))  # the first match in the file's text
    if not m:  # if: nothing matches
        raise KeyError(f"{pattern} not found in {path}")  # stops: the source changed
    return m.group(1)  # the captured literal, as text


def function_literal(path: str, func: str, var: str, index: int) -> float:  # function: the index-th numeric literal assigned to var inside func
    tree = ast.parse((PROJECT / path).read_text(encoding="utf-8"))  # the file's syntax tree
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)  # generator with next(): the function definition
    for node in ast.walk(fn):  # for loop over every node inside the function
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == var for t in node.targets):  # if: the assignment to var
            consts = [n for n in ast.walk(node.value) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))]  # list comprehension: the numeric literal nodes (ast.walk visits breadth-first, not left to right)
            consts.sort(key=lambda n: (n.lineno, n.col_offset))  # list sort with a lambda key: puts them in source order, left to right
            return consts[index].value  # the requested literal, counted in source order
    raise KeyError(f"{var} not assigned in {func}")  # stops when the variable is missing





