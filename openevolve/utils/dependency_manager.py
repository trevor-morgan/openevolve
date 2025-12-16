"""
Automated dependency management for generated code.

This module provides functionality to detect imported packages in Python code
and automatically install them if they are missing.
"""

import ast
import importlib
import importlib.util
import logging
import subprocess
import sys
import threading

logger = logging.getLogger(__name__)

# Lock for installation to prevent race conditions in threaded environments
_install_lock = threading.Lock()

# Standard library modules to ignore (Python 3.10+)
# This list is not exhaustive but covers common modules
STD_LIB = {
    "abc",
    "argparse",
    "array",
    "ast",
    "asyncio",
    "base64",
    "bisect",
    "builtins",
    "bz2",
    "calendar",
    "cmath",
    "collections",
    "concurrent",
    "contextlib",
    "copy",
    "csv",
    "ctypes",
    "dataclasses",
    "datetime",
    "decimal",
    "difflib",
    "dis",
    "doctest",
    "email",
    "enum",
    "errno",
    "faulthandler",
    "fcntl",
    "filecmp",
    "fnmatch",
    "fractions",
    "functools",
    "gc",
    "getopt",
    "getpass",
    "glob",
    "graphlib",
    "gzip",
    "hashlib",
    "heapq",
    "hmac",
    "html",
    "http",
    "imaplib",
    "importlib",
    "inspect",
    "io",
    "ipaddress",
    "itertools",
    "json",
    "keyword",
    "lib2to3",
    "linecache",
    "locale",
    "logging",
    "lzma",
    "math",
    "mmap",
    "modulefinder",
    "multiprocessing",
    "netrc",
    "nntplib",
    "numbers",
    "operator",
    "os",
    "pathlib",
    "pickle",
    "pickletools",
    "pkgutil",
    "platform",
    "plistlib",
    "poplib",
    "posix",
    "pprint",
    "profile",
    "pstats",
    "pty",
    "pwd",
    "py_compile",
    "pyclbr",
    "pydoc",
    "queue",
    "quopri",
    "random",
    "re",
    "readline",
    "reprlib",
    "resource",
    "rlcompleter",
    "runpy",
    "sched",
    "secrets",
    "select",
    "selectors",
    "shelve",
    "shlex",
    "shutil",
    "signal",
    "site",
    "smtpd",
    "smtplib",
    "sndhdr",
    "socket",
    "socketserver",
    "sqlite3",
    "ssl",
    "stat",
    "statistics",
    "string",
    "stringprep",
    "struct",
    "subprocess",
    "sunau",
    "symbol",
    "symtable",
    "sys",
    "sysconfig",
    "syslog",
    "tabnanny",
    "tarfile",
    "telnetlib",
    "tempfile",
    "termios",
    "textwrap",
    "threading",
    "time",
    "timeit",
    "tkinter",
    "token",
    "tokenize",
    "trace",
    "traceback",
    "tracemalloc",
    "tty",
    "turtle",
    "turtledemo",
    "types",
    "typing",
    "unicodedata",
    "unittest",
    "urllib",
    "uu",
    "uuid",
    "venv",
    "warnings",
    "wave",
    "weakref",
    "webbrowser",
    "wsgiref",
    "xdrlib",
    "xml",
    "xmlrpc",
    "zipapp",
    "zipfile",
    "zipimport",
    "zlib",
    "zoneinfo",
}


class DependencyManager:
    """Manages dependencies for generated code."""

    def __init__(self, auto_install: bool = False):
        self.auto_install = auto_install
        self._installed_cache: set[str] = set()

    def check_and_install(self, code: str) -> None:
        """
        Analyze code for imports and install missing dependencies.

        Args:
            code: Python code string to analyze
        """
        if not self.auto_install:
            return

        imports = self._extract_imports(code)
        missing = self._filter_missing(imports)

        if missing:
            self._install_packages(missing)

    def _extract_imports(self, code: str) -> set[str]:
        """Extract top-level package names from imports."""
        imports = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.add(name.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split(".")[0])
        except SyntaxError:
            # Code might be partial or invalid, try regex as fallback
            import re

            # Match "import X" and "from X import Y"
            matches = re.findall(r"^\s*(?:import|from)\s+(\w+)", code, re.MULTILINE)
            imports.update(matches)

        return imports

    def _filter_missing(self, packages: set[str]) -> set[str]:
        """Filter out installed packages and standard library."""
        missing = set()
        for pkg in packages:
            if pkg in STD_LIB:
                continue
            if pkg in self._installed_cache:
                continue

            # Check if importable
            if self._is_installed(pkg):
                self._installed_cache.add(pkg)
                continue

            missing.add(pkg)
        return missing

    def _is_installed(self, package_name: str) -> bool:
        """Check if a package is importable."""
        try:
            # Use importlib to check without actually importing
            spec = importlib.util.find_spec(package_name)
            return spec is not None
        except (ImportError, ValueError):
            return False

    def _install_packages(self, packages: set[str]) -> None:
        """Install packages via pip."""
        if not packages:
            return

        with _install_lock:
            # Re-check inside lock to avoid redundant installs
            to_install = [p for p in packages if not self._is_installed(p)]

            if not to_install:
                return

            logger.info(f"Auto-installing missing dependencies: {', '.join(to_install)}")

            try:
                # Install packages
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", *to_install],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )

                # Update cache
                self._installed_cache.update(to_install)

                # Invalidate import caches so newly installed modules are found
                importlib.invalidate_caches()

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install dependencies {to_install}: {e}")
                # We log but don't raise, allowing the code to fail naturally later with ImportError
