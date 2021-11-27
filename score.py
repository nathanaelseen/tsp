"""
Copyright: MIT License Soeren Hougaard Mulvad <shmulvad@gmail.com> 2021
This program tests and scores your solutions for the various problems you have
in your test data.
You need to install requirements beforehand:
$ pip install click sty
The program can be run by executing this file. E.g.
$ python score.py --help  # See how to run
$ python score.py --program=tsp.cpp --problem-type=tsp --test-glob='./tests/*.txt' --timeout=5 -j 4 -max 50
Currently, only Python and C++ and programs as well as executables are supported.
If you use any other file type, behavior is undefined.
The program supports scoring for TSP and MWVC. If you need any other problem type,
you need to implement a custom `ProblemChecker` class and add it to the `CHECKERS`
dictionary. Depending on the nature of the problem, you might also need to
change the scoring function.
The program assumes your test files are in the following format:
```
[OPT] [NAIVE]
[REMAINDER OF FILE IS ORIGINAL PROBLEM INPUT]
```
"""
import logging
import re
import subprocess
import sys
import math
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from logging.handlers import RotatingFileHandler
from tempfile import TemporaryFile
from typing import List, Optional, Tuple, Union

import click
from sty import fg
fg.pink = fg(201)

MAX_LOG_BYTES = 1024 * 1024 * 10  # 10 MB
MAX_ERROR_OUTPUT = 100

DEFAULT_TIMEOUT = 100  # seconds before test killed
DEFAULT_TEST = './tests/*.txt'
DEFAULT_VERBOSE = 2
DEFAULT_N_JOBS = 4
DEFAULT_MAX_SCORE = 50
INDENT = '    '

GCC_ARGS = ['g++', '-std=c++17']  # The gcc compiler args for cpp programs
PYTHON = sys.executable  # The Python interpreter for running Python programs

INF = float('inf')


def get_logger():
    """Get a new logger instance"""
    logger = logging.getLogger('scoring')
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    f_handler = RotatingFileHandler('score.log', maxBytes=MAX_LOG_BYTES)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger


logger = get_logger()


def set_log_level(logger: logging.Logger, verbose: int):
    """Sets log level based on verbosity level"""
    log_level = logging.INFO if verbose == 1 else logging.DEBUG
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)


class TestVerdict:
    def __init__(self,
                 cost: Optional[float] = None,
                 err_msg: Optional[str] = None):
        self.cost = cost
        self.err_msg = err_msg
        self.score = None
        self.in_file = None

    @property
    def ok(self) -> bool:
        """Whether the test case succeded or some error occured"""
        return self.err_msg is None

    def compute_score(self, naive: int, opt: int):
        """
        Sets the score for the verdict's cost as per the Kattis specifications
        """
        if not self.ok:
            return

        if naive == opt:
            self.score = 1.0 if self.cost == opt else 0.0
        else:
            relative_diff = (self.cost - opt) / (naive - opt)
            self.score = 0.02**max(relative_diff, 0)

    def set_in_file(self, in_file: str):
        """Sets which `in_file` was used as test case"""
        self.in_file = in_file

    def get_msg(self, max_width: int) -> str:
        """
        Returns a string representation of the test verdict, padding the
        leading part to a maximum width of `max_width` if needed
        """
        if self.ok and self.score >= 0.999:
            color = fg.green
        elif self.ok and self.score >= 0.8:
            color = fg.yellow
        elif self.ok:
            color = fg.pink
        else:
            color = fg.red

        msg = (f'[ok] - {self.score:.3f}'
               if self.ok
               else f'[failed] {self.err_msg}')
        return f'{self.in_file.ljust(max_width)} {color}{msg}{fg.rs}'


class ProblemChecker(ABC):
    def perform_check(self, answer: List[str],
                      problem_input: List[str]) -> TestVerdict:
        try:
            return self.check(answer, problem_input)
        except Exception:
            answer_str = ',   '.join(answer)
            answer_shortened = (answer_str[:MAX_ERROR_OUTPUT] + '...'
                                if len(answer_str) > MAX_ERROR_OUTPUT
                                else answer_str)
            if answer_shortened == '':
                answer_shortened = '<empty>'
            return TestVerdict(err_msg=f'WA (Received {answer_shortened})')

    @abstractmethod
    def check(self, answer: List[str], problem_input: List[str]) -> TestVerdict:
        pass


class TSPChecker(ProblemChecker):
    def compute_cost(self,
                     coords: List[Tuple[float, float]],
                     tour: List[int]) -> int:
        """
        Computes the cost for an array of coordinates and a list of the tour
        """
        cost = 0
        for idx1, idx2 in zip(tour, tour[1:] + [tour[0]]):
            (x1, y1), (x2, y2) = coords[idx1], coords[idx2]
            cost += round(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))
        return cost

    def check(self, answer: List[str], problem_input: List[str]) -> TestVerdict:
        """
        Custom function to check that TSP tour is valid and compute cost of it
        """
        tour = list(map(int, answer))
        n, *coords_raw = problem_input
        n = int(n)

        is_correct_len = len(tour) == len(set(tour)) == n
        is_correct_bounds = min(tour) == 0 and max(tour) == n - 1
        is_valid = is_correct_len and is_correct_bounds
        if not is_valid:
            err_msg = 'WA'
            if not is_correct_len:
                err_msg += (f' (Incorrect length, n={n} but {len(tour)} in sol,'
                            + f' {len(set(tour))} after removing duplicates)')
            if not is_correct_bounds:
                err_msg += (f' (Incorrect bounds - min(tour) = {min(tour)} != 0'
                            + f' or max(tour) = {max(tour)} != {n - 1})')
            return TestVerdict(err_msg=err_msg)

        coords = [tuple(float(val) for val in line.split())
                  for line in coords_raw]
        return TestVerdict(cost=self.compute_cost(coords, tour))


class MWVCChecker(ProblemChecker):
    def check(answer: List[str], problem_input: List[int]) -> TestVerdict:
        """
        Custom function to check that vertex cover is valid and compute cost of
        vertex cover
        """
        cost_told, vc_line = answer
        _, weights_raw, *edges = problem_input
        cost_told = int(cost_told)
        answer_vc = set(map(int, vc_line.split()))
        weigths = [int(w) for w in weights_raw.split()]
        for idx, edge in enumerate(edges):
            u, v = map(int, edge.split())
            if not (u in answer_vc or v in answer_vc):
                return TestVerdict(err_msg=f'WA (Not a valid VC, edge idx {idx} uncovered)')

        cost = sum(weigths[v] for v in answer_vc)
        if cost != cost_told:
            return TestVerdict(err_msg=f'WA (Incorrect VC cost - got {cost_told}, true is {cost})')

        return TestVerdict(cost=cost)


# Add a checker here if you're defining implementing a new problem type
CHECKERS = {
    'tsp': TSPChecker,
    'mwvc': MWVCChecker
}


# It should not be necessary to modify the file below this line
# -----------------------------------------------------------------------------


def purify(lines: List[str]) -> List[str]:
    """Remove redundant white spaces from list of lines"""
    lines_stripped = [re.sub(r'\s+', ' ', line).strip() for line in lines]
    return [line for line in lines_stripped if line]


def load_text(file: Union[str, TemporaryFile]) -> List[str]:
    """Load and purify text file"""
    if isinstance(file, str):
        with open(file) as f:
            return purify(f.readlines())

    return purify(file.readlines())


def get_stats_str(num_passed: int, num_tests: int, score: float,
                  max_score: int) -> str:
    """
    Construct string of the number of passed tests and the normalized score
    """
    score_normalized = score * max_score / num_tests
    msg = f'{num_passed}/{num_tests} ({score_normalized:.2f}/{max_score})'
    dashes = '-' * len(msg)

    out_msg = '\n'.join([
        '\nTests passed (score):\n',
        INDENT + dashes,
        INDENT + msg,
        INDENT + dashes
    ])
    return out_msg


def run_test(program: str, test_in: str, timeout: int,
             problem_type: str, is_python: bool) -> TestVerdict:
    """
    Runs program on test input with the given timeout and returns the result
    """
    opt, naive, exception = None, None, ''
    with TemporaryFile('w+') as answer_file, \
         TemporaryFile('w+') as error_file, \
         open(test_in, 'r') as in_file:
        try:
            opt_naive, *problem_input_lines = load_text(in_file)
            opt, naive = map(int, opt_naive.split())
            problem_input = '\n'.join(problem_input_lines)
            cmd = [PYTHON, program] if is_python else [f'./{program}']
            subprocess.run(
                cmd,
                input=problem_input,
                stdout=answer_file,
                stderr=error_file,
                timeout=timeout,
                text=True
            )
        except Exception as e:
            exception = str(e)

        answer_file.seek(0)
        error_file.seek(0)

        answer = load_text(answer_file)
        error = load_text(error_file)

        if exception:
            error.append(exception)

    if error:
        verdict = TestVerdict(err_msg=', '.join(error))
    else:
        problem_checker = CHECKERS[problem_type]()
        verdict = problem_checker.perform_check(answer, problem_input_lines)
        verdict.compute_score(naive, opt)

    verdict.set_in_file(test_in)
    return verdict


def run_test_wrapper(args) -> TestVerdict:
    """
    Wrapper function since function running in parallel can only take one arg
    """
    return run_test(*args)


def try_compile_cpp(program: str):
    """Tries to compile a cpp program. Exits in case of any errors occuring"""
    logger.debug(f'Compiling {program}.cpp...')
    try:
        cmd = [*GCC_ARGS, '-o', f'{program}.o', f'{program}.cpp']
        ret_val = subprocess.call(cmd, text=True)
        if ret_val != 0:
            exit()
    except Exception as e:
        logger.exception(e)
        exit()


def handle_input_program(program: str) -> Tuple[str, str]:
    """
    Handles input program. If it is a cpp program, compiles it.
    Returns a tuple of the program name and the extension
    """
    if program.endswith('.py'):
        return program[:-3], '.py'

    if program.endswith('.cpp'):
        try_compile_cpp(program[:-4])
        return program[:-4], '.o'

    # Base case, e.g. if executable is passed in
    return program, ''


def test_all(program: str, problem_type: str, test_glob: str, timeout: int,
             njobs: int, max_score: int) -> None:
    """Run all test cases and report the score"""
    program, file_ext = handle_input_program(program)
    is_python = file_ext == '.py'

    test_files = sorted(glob(test_glob))
    if not test_files:
        logger.error(f'No test files found by pattern {test_glob}')
        exit()

    max_width = max(map(len, test_files))
    num_passed, total_score = 0, 0
    args = tuple((program + file_ext, test_in, timeout, problem_type, is_python)
                 for test_in in test_files)

    # Run all test cases in njobs parallel jobs
    with ProcessPoolExecutor(max_workers=njobs) as executor:
        for verdict in executor.map(run_test_wrapper, args):
            logger.info(f'{program} {verdict.get_msg(max_width)}')
            if verdict.ok:
                num_passed += 1
                total_score += verdict.score

    # Log final score
    score_str = get_stats_str(num_passed, len(test_files), total_score, max_score)
    logger.info(score_str)


@click.command()
@click.option('-p', '--program', required=True,
              help='Program to run, e.g. tsp.cpp')
@click.option('-pt', '--problem-type', required=True,
              type=click.Choice(CHECKERS, case_sensitive=False),
              help='The problem type to test')
@click.option('-v', '--verbose', default=DEFAULT_VERBOSE, type=int,
              help='Verbosity level',
              show_default=True)
@click.option('-tg', '--test-glob', default=DEFAULT_TEST, type=str,
              help='The glob pattern for test cases',
              show_default=True)
@click.option('-t', '--timeout', default=DEFAULT_TIMEOUT, type=int,
              help='Time limit in seconds for each test case',
              show_default=True)
@click.option('-j', '--jobs', default=DEFAULT_N_JOBS, type=int,
              help='Number of test cases to run in parallel',
              show_default=True)
@click.option('-max', '--max-score', default=DEFAULT_MAX_SCORE, type=int,
              help='The maximum obtainable score',
              show_default=True)
def main(program, problem_type, verbose, test_glob, timeout, jobs, max_score):
    set_log_level(logger, verbose)
    logger.info(f'Testing {program}')
    try:
        test_all(program, problem_type, test_glob, timeout, jobs, max_score)
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    main()
