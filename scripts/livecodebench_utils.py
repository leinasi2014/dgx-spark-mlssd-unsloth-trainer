#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Part of the codes are from https://github.com/NovaSky-AI/SkyThought/blob/main/skythought/tools/util/livecodebench/testing_util.py
"""

from __future__ import annotations

import ast
import base64
import copy
import faulthandler
import io
import json
import multiprocessing
import pickle
import sys
import time
import zlib
from decimal import Decimal
from types import ModuleType
from typing import Any
from unittest.mock import mock_open, patch

import numpy as np

BASE_IMPORTS = """from itertools import accumulate, chain, combinations, count, permutations, product, groupby, islice, repeat
from copy import deepcopy
from string import ascii_lowercase
from math import floor, log2, log10, sqrt, comb, gcd, ceil, inf, isqrt
from collections import defaultdict, deque, Counter
from bisect import bisect, bisect_left, bisect_right, insort
from heapq import heappush, heappop, heapify, merge
from functools import reduce, cache, lru_cache
from random import randrange, shuffle
from operator import itemgetter, sub
from re import search as re_search
from os.path import commonprefix
from typing import List, Tuple, Dict, Set, Optional, Union, Any, Callable, Iterable, Iterator, Generator
import copy
import string
import math
import collections
import bisect
import heapq
import functools
import random
import itertools
import operator
import re
import numpy as np
import pandas as pd
from math import log, prod
from collections import deque, defaultdict, Counter, OrderedDict
from itertools import accumulate, permutations, combinations, product, groupby, islice, chain, repeat, zip_longest, cycle
from functools import lru_cache, reduce, partial
from operator import iand
import sys
"""


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


class MockBuffer:
    def __init__(self, inputs: str):
        self._buffer = io.BytesIO(inputs.encode('utf-8'))

    def read(self, *args):
        return self._buffer.read(*args)

    def readline(self, *args):
        return self._buffer.readline(*args)


class MockStdinWithBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = io.StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self._stringio.read(*args)

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self._stringio.readlines(*args)

    def __iter__(self):
        return iter(self._stringio)

    def __next__(self):
        return next(self._stringio)

    def __getattr__(self, name):
        return getattr(self._stringio, name)


def get_stripped_lines(val: str) -> list[str]:
    val = val.strip()
    return [val_line.strip() for val_line in val.split('\n')]


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []
    return True, decimal_line


def compare_strings_with_decimal_fallback(prediction_str: str, expected_str: str) -> bool:
    stripped_prediction_lines = get_stripped_lines(prediction_str)
    stripped_expected_lines = get_stripped_lines(expected_str)
    if len(stripped_prediction_lines) != len(stripped_expected_lines):
        return False
    for stripped_pred_line, stripped_exp_line in zip(stripped_prediction_lines, stripped_expected_lines):
        if stripped_pred_line == stripped_exp_line:
            continue
        success, decimal_pred_line = convert_line_to_decimals(stripped_pred_line)
        if not success:
            return False
        success, decimal_exp_line = convert_line_to_decimals(stripped_exp_line)
        if not success:
            return False
        if decimal_pred_line == decimal_exp_line:
            continue
        return False
    return True


def reliability_guard() -> None:
    faulthandler.disable()
    import builtins
    import os
    import shutil
    import subprocess

    builtins.exit = None
    builtins.quit = None
    os.environ['OMP_NUM_THREADS'] = '1'
    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None
    subprocess.Popen = None  # type: ignore
    sys.modules['ipdb'] = None
    sys.modules['joblib'] = None
    sys.modules['resource'] = None
    sys.modules['psutil'] = None
    sys.modules['tkinter'] = None


def has_test_type(tests: str, test_type: str) -> bool:
    test_list = json.loads(tests)
    for test in test_list:
        if test.get('testtype') == test_type:
            return True
    return False


def translate_private_test_cases(encoded_data: str) -> list[dict[str, Any]]:
    decoded_data = base64.b64decode(encoded_data)
    decompressed_data = zlib.decompress(decoded_data)
    original_data = pickle.loads(decompressed_data)
    test_cases = json.loads(original_data)
    return test_cases


def map_to_example(row: dict[str, Any]) -> dict[str, Any]:
    metadata_raw = row.get('metadata', '{}')
    try:
        metadata = json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
    except json.JSONDecodeError:
        metadata = {}
    return {
        'prompt': row['question_content'],
        'test': row['private_test_cases'],
        'entry_point': row['starter_code'],
        'task_id': row['question_id'],
        'is_stdin': has_test_type(row['public_test_cases'], 'stdin'),
        'public_test_cases': row['public_test_cases'],
        'difficulty': row['difficulty'],
        'metadata': metadata,
    }


def post_process_code(code: str) -> str:
    code = code.split('</code>')[0]
    code = code.replace('```python', '')
    code = code.split('```')[0]
    code = code.replace('<code>', '')
    return code


def parse_function_name_from_starter_code(starter_code: str) -> str | None:
    try:
        code_to_parse = starter_code
        if not code_to_parse.strip().endswith(('pass', '...', 'return')):
            lines = code_to_parse.rstrip().split('\n')
            if lines:
                last_line = lines[-1]
                if last_line.rstrip().endswith(':'):
                    indent = len(last_line) - len(last_line.lstrip()) + 4
                    code_to_parse = code_to_parse + '\n' + ' ' * indent + 'pass'
        tree = ast.parse(code_to_parse)
        fn = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                fn = node.name
        return fn
    except Exception:
        return None


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = ast.unparse(astree.body[:-1]) + '\n' + ast.unparse(last_block.body)  # type: ignore
    except Exception:
        pass
    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name='wrapped_function',
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        return BASE_IMPORTS + '\n' + ast.unparse(import_stmts) + '\n' + ast.unparse(function_ast)  # type: ignore
    except Exception:
        return code


def compile_code(code: str) -> ModuleType | None:
    try:
        tmp_sol = ModuleType('tmp_sol', '')
        exec(code, tmp_sol.__dict__)
        if 'class Solution' in code:
            compiled_sol = tmp_sol.Solution()
        else:
            compiled_sol = tmp_sol
        return compiled_sol
    except Exception:
        return None


def get_function(compiled_sol: Any, fn_name: str) -> Any:
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return None


def call_method(method: Any, inputs: Any) -> Any:
    if isinstance(inputs, list):
        inputs = '\n'.join(inputs)
    mock_stdin = MockStdinWithBuffer(inputs)

    @patch('builtins.open', mock_open(read_data=inputs))
    @patch('sys.stdin', mock_stdin)
    def _inner_call_method(_method: Any):
        try:
            return _method()
        except SystemExit:
            return None

    return _inner_call_method(method)


def prepare_test_input_output_std(test_case: dict[str, Any]) -> tuple[str, str]:
    return test_case['input'], test_case['output'].strip()


def run_test_func(completion: str, is_extracted: bool, test_input: Any, test_output: Any, func_name: str) -> tuple[bool, Any]:
    assert func_name is not None, 'func_name must be provided'
    namespace: dict[str, Any] = {}
    exec(completion, namespace)
    is_class_based = 'class Solution:' in completion or 'class Solution(' in completion
    output = io.StringIO()
    sys.stdout = output
    try:
        if is_class_based:
            solution_instance = namespace['Solution']()
            callable_func = getattr(solution_instance, func_name)
        else:
            callable_func = namespace[func_name]
        if not is_extracted:
            if isinstance(test_input, dict):
                prediction = callable_func(**test_input)
            else:
                prediction = callable_func(test_input)
        else:
            prediction = callable_func(*test_input)
        if isinstance(prediction, tuple):
            prediction = list(prediction)
        prediction_str = str(prediction) if not isinstance(prediction, str) else prediction
        expected_str = str(test_output) if not isinstance(test_output, str) else test_output
        return compare_strings_with_decimal_fallback(prediction_str, expected_str), prediction
    except Exception as exc:
        error_msg = f'Error: {str(exc)}' if not is_extracted else str(exc)
        return False, error_msg
    finally:
        sys.stdout = sys.__stdout__


def run_test_std(completion: str, test_input: str, test_output: str) -> tuple[bool, str]:
    completion = clean_if_name(completion)
    completion = make_function(completion)
    compiled_sol = compile_code(completion)
    if compiled_sol is None:
        return False, 'Compilation failed'
    method = get_function(compiled_sol, 'wrapped_function')
    if method is None:
        return False, 'Could not find wrapped_function'
    with Capturing() as captured_output:
        try:
            call_method(method, test_input)
        except Exception as exc:
            return False, f'Runtime error: {exc}'
    prediction = captured_output[0] if captured_output else ''
    return compare_strings_with_decimal_fallback(prediction, test_output), prediction.strip()


def prepare_test_input_output_functional(test_case: dict[str, Any], is_extracted: bool) -> tuple[Any, Any]:
    if not is_extracted:
        return test_case['input'], test_case['output']
    input_str = test_case['input']
    expected_output = test_case['output'].strip()
    inputs = []
    if '=' in input_str:
        parts = input_str.split(',') if ',' in input_str else [input_str]
        for part in parts:
            key, value = map(str.strip, part.split('='))
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    value = value.strip('"')
            inputs.append(value)
    else:
        for line in input_str.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith('"') and line.endswith('"'):
                inputs.append(line.strip('"'))
                continue
            if line.startswith('[') and line.endswith(']'):
                inputs.append(json.loads(line))
                continue
            try:
                inputs.append(int(line))
            except ValueError:
                try:
                    inputs.append(float(line))
                except ValueError:
                    inputs.append(line)
    try:
        expected_output = json.loads(expected_output)
    except json.JSONDecodeError:
        expected_output = expected_output.strip()
    return inputs, expected_output


def run_tests_for_one_example(problem: dict[str, Any], test_cases: list[dict[str, Any]], completion: str, result_list: Any, is_extracted: bool) -> None:
    reliability_guard()
    completion = BASE_IMPORTS + '\n' + completion
    func_name = None
    test_type = test_cases[0]['testtype']
    if test_type == 'functional':
        metadata = problem.get('metadata', {})
        func_name = metadata.get('func_name')
        if not func_name and 'entry_point' in problem:
            func_name = parse_function_name_from_starter_code(problem['entry_point'])
    for test_case in test_cases:
        output_error = ''
        output_value = ''
        try:
            time_start = time.time()
            if test_type == 'functional':
                test_input, test_output = prepare_test_input_output_functional(test_case, is_extracted)
                passed, output_value = run_test_func(completion, is_extracted, copy.deepcopy(test_input), copy.deepcopy(test_output), func_name)
            else:
                test_input, test_output = prepare_test_input_output_std(test_case)
                passed, output_value = run_test_std(completion, copy.deepcopy(test_input), copy.deepcopy(test_output))
            time_elapsed = time.time() - time_start
            if not passed:
                output_error = f'For test input: {test_input}. Expected output is: {test_output}, but got: {output_value}.'
        except Exception as exc:
            passed = False
            output_error = f'For test input: {test_input}. Expected output is: {test_output}, but got error: {exc}.'
            output_value = f'Error: {exc}.'
            time_elapsed = float('inf')
        if output_error == '':
            output_error = f'For test input: {test_input}. Expected output is: {test_output}, your solution correctly passes this test with output {output_value}.'
        result_list.append((passed, output_error, output_value, time_elapsed))
        if not passed:
            return


def lcb_run(problem: dict[str, Any], completion: str, timeout: float, is_extracted: bool) -> Any:
    test_cases = problem['test']
    manager = multiprocessing.Manager()
    result = manager.list()
    process = multiprocessing.Process(target=run_tests_for_one_example, args=(problem, test_cases, completion, result, is_extracted))
    process.start()
    process.join(timeout=(timeout + 1) * len(test_cases) + 5)
    if process.is_alive():
        process.kill()
    for _ in range(len(test_cases) - len(result)):
        result.append((False, 'Time out!.', 'Error: Time out!', float('inf')))
    return result


def estimate_pass_at_k(num_samples: Any, num_correct: Any, k: int) -> np.ndarray:
    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    import itertools

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)
    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def compute_metrics_from_results(results: dict[str, list[list[int]]], k_list: list[int] | None = None) -> dict[str, Any]:
    total = []
    correct = []
    task_ids = []
    k_values = k_list or [1, 5]
    for task_id, res in results.items():
        all_correct = []
        for generation in res:
            gen = np.array(generation)
            all_correct.append(np.all(gen > 0))
        task_ids.append(task_id)
        total.append(len(all_correct))
        correct.append(sum(all_correct))
    total_np = np.array(total)
    correct_np = np.array(correct)
    detail_pass_at_k = {
        f'pass@{k}': estimate_pass_at_k(total_np, correct_np, k).tolist()
        for k in k_values
        if (total_np >= k).all()
    }
    pass_at_k = {
        f'pass@{k}': estimate_pass_at_k(total_np, correct_np, k).mean()
        for k in k_values
        if (total_np >= k).all()
    }
    pass_at_k['detail'] = {metric: dict(zip(task_ids, values)) for metric, values in detail_pass_at_k.items()}
    return pass_at_k
