"""
Code Instrumenter for Heisenberg Engine

Auto-instruments Python code to capture execution traces, intermediate values,
and other data that can be analyzed by probes to discover hidden variables.

Instrumentation Levels:
- minimal: Basic function call counts and timing
- standard: Add variable tracking and control flow
- comprehensive: Full execution trace with all intermediate values
"""

import ast
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InstrumentationResult:
    """Result of code instrumentation"""

    instrumented_code: str
    original_code: str
    instrumentation_level: str
    tracked_variables: list[str]
    injection_points: int
    success: bool
    error: str | None = None


class TraceCollector:
    """
    Runtime trace collector injected into instrumented code.

    This class is serialized and injected into the instrumented code
    to collect execution traces.
    """

    COLLECTOR_CODE = '''
class __TraceCollector__:
    """Collects execution traces for Heisenberg Engine analysis"""

    def __init__(self):
        self.traces = {
            "function_calls": [],
            "variable_assignments": {},
            "loop_iterations": {},
            "branch_decisions": [],
            "intermediate_values": {},
            "timing": {},
            "counters": {},
            "errors": [],
        }
        self._start_time = None

    def start(self):
        import time
        self._start_time = time.time()

    def stop(self):
        import time
        if self._start_time:
            self.traces["timing"]["total_time"] = time.time() - self._start_time

    def log_call(self, func_name, args_repr=""):
        import time
        self.traces["function_calls"].append({
            "name": func_name,
            "args": args_repr[:100],
            "time": time.time() - (self._start_time or 0)
        })

    def log_assignment(self, var_name, value):
        if var_name not in self.traces["variable_assignments"]:
            self.traces["variable_assignments"][var_name] = []
        # Store value representation (avoid storing huge objects)
        try:
            if isinstance(value, (int, float, bool, str)):
                val_repr = value
            elif isinstance(value, (list, tuple)) and len(value) < 10:
                val_repr = list(value)
            elif hasattr(value, "shape"):  # numpy array
                val_repr = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                val_repr = str(type(value).__name__)
        except:
            val_repr = "unknown"
        self.traces["variable_assignments"][var_name].append(val_repr)
        # Keep only last 100 assignments per variable
        if len(self.traces["variable_assignments"][var_name]) > 100:
            self.traces["variable_assignments"][var_name] = self.traces["variable_assignments"][var_name][-100:]

    def log_intermediate(self, name, value):
        if name not in self.traces["intermediate_values"]:
            self.traces["intermediate_values"][name] = []
        try:
            if isinstance(value, (int, float)):
                self.traces["intermediate_values"][name].append(float(value))
            elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value[:10]):
                self.traces["intermediate_values"][name].extend([float(x) for x in value[:10]])
        except:
            pass
        # Keep only last 100 values
        if len(self.traces["intermediate_values"][name]) > 100:
            self.traces["intermediate_values"][name] = self.traces["intermediate_values"][name][-100:]

    def log_loop_iteration(self, loop_id):
        if loop_id not in self.traces["loop_iterations"]:
            self.traces["loop_iterations"][loop_id] = 0
        self.traces["loop_iterations"][loop_id] += 1

    def log_branch(self, branch_id, taken):
        self.traces["branch_decisions"].append({"id": branch_id, "taken": taken})
        if len(self.traces["branch_decisions"]) > 1000:
            self.traces["branch_decisions"] = self.traces["branch_decisions"][-1000:]

    def increment_counter(self, name):
        if name not in self.traces["counters"]:
            self.traces["counters"][name] = 0
        self.traces["counters"][name] += 1

    def log_error(self, error_type, message, location=""):
        self.traces["errors"].append({
            "type": error_type,
            "message": str(message)[:200],
            "location": location
        })

    def get_traces(self):
        return self.traces

__heisenberg_trace__ = __TraceCollector__()
__heisenberg_trace__.start()
'''

    COLLECTOR_FINALIZE = """
__heisenberg_trace__.stop()
__heisenberg_artifacts__ = __heisenberg_trace__.get_traces()
"""


class CodeInstrumenter:
    """
    Auto-instruments Python code to capture execution traces.

    The instrumenter parses Python code and injects trace collection
    calls at strategic points to capture data that probes can analyze.

    Usage:
        instrumenter = CodeInstrumenter()
        result = instrumenter.instrument(code, level="standard")
        if result.success:
            # Execute result.instrumented_code
            # Access traces via __heisenberg_artifacts__
    """

    def __init__(self):
        self.instrumentation_count = 0

    def instrument(
        self,
        code: str,
        level: str = "standard",
        evolve_block_only: bool = True,
    ) -> InstrumentationResult:
        """
        Instrument Python code to capture execution traces.

        Args:
            code: The Python code to instrument
            level: Instrumentation level ("minimal", "standard", "comprehensive")
            evolve_block_only: Only instrument code within EVOLVE-BLOCK markers

        Returns:
            InstrumentationResult with instrumented code
        """
        try:
            # Parse the code
            tree = ast.parse(code)

            # Find evolve block if requested
            if evolve_block_only:
                code, block_start, _block_end = self._extract_evolve_block(code)
                if block_start == -1:
                    # No evolve block, instrument entire code
                    evolve_block_only = False
                else:
                    # Re-parse just the evolve block
                    try:
                        tree = ast.parse(code)
                    except SyntaxError:
                        # If block isn't valid Python alone, instrument full code
                        code = code
                        tree = ast.parse(code)
                        evolve_block_only = False

            # Apply instrumentation based on level
            if level == "minimal":
                transformer = MinimalInstrumenter()
            elif level == "standard":
                transformer = StandardInstrumenter()
            elif level == "comprehensive":
                transformer = ComprehensiveInstrumenter()
            else:
                raise ValueError(f"Unknown instrumentation level: {level}")

            instrumented_tree = transformer.visit(tree)
            ast.fix_missing_locations(instrumented_tree)

            # Generate instrumented code
            try:
                import astor

                instrumented_code = astor.to_source(instrumented_tree)
            except ImportError:
                # Fall back to ast.unparse (Python 3.9+)
                instrumented_code = ast.unparse(instrumented_tree)

            # Add trace collector
            full_code = (
                TraceCollector.COLLECTOR_CODE
                + "\n\n"
                + instrumented_code
                + "\n\n"
                + TraceCollector.COLLECTOR_FINALIZE
            )

            return InstrumentationResult(
                instrumented_code=full_code,
                original_code=code,
                instrumentation_level=level,
                tracked_variables=transformer.tracked_variables,
                injection_points=transformer.injection_count,
                success=True,
            )

        except SyntaxError as e:
            logger.warning(f"Code instrumentation failed (syntax error): {e}")
            return InstrumentationResult(
                instrumented_code=code,
                original_code=code,
                instrumentation_level=level,
                tracked_variables=[],
                injection_points=0,
                success=False,
                error=f"Syntax error: {e}",
            )

        except Exception as e:
            logger.warning(f"Code instrumentation failed: {e}")
            return InstrumentationResult(
                instrumented_code=code,
                original_code=code,
                instrumentation_level=level,
                tracked_variables=[],
                injection_points=0,
                success=False,
                error=str(e),
            )

    def _extract_evolve_block(self, code: str) -> tuple:
        """Extract code within EVOLVE-BLOCK markers"""
        start_marker = "# EVOLVE-BLOCK-START"
        end_marker = "# EVOLVE-BLOCK-END"

        start_idx = code.find(start_marker)
        end_idx = code.find(end_marker)

        if start_idx == -1 or end_idx == -1:
            return code, -1, -1

        # Extract the block (including some context)
        block_start = code.find("\n", start_idx) + 1
        block_end = end_idx

        block_code = code[block_start:block_end].strip()

        return block_code, block_start, block_end

    def extract_traces(self, execution_output: str) -> dict[str, Any]:
        """
        Extract traces from execution output.

        Args:
            execution_output: The stdout from executing instrumented code

        Returns:
            Dictionary of extracted traces
        """
        # Look for JSON trace data in output
        import json

        # Try to find JSON block
        json_match = re.search(
            r"__heisenberg_artifacts__\s*=\s*(\{.*\})", execution_output, re.DOTALL
        )
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object
        json_match = re.search(r'\{[^{}]*"traces"[^{}]*\}', execution_output)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {"error": "Could not extract traces from output"}


class BaseInstrumenter(ast.NodeTransformer):
    """Base class for code instrumenters"""

    def __init__(self):
        self.tracked_variables: list[str] = []
        self.injection_count = 0
        self._loop_counter = 0
        self._branch_counter = 0

    def _make_trace_call(self, method: str, *args) -> ast.Expr:
        """Create a trace collector method call"""
        self.injection_count += 1

        call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="__heisenberg_trace__", ctx=ast.Load()),
                attr=method,
                ctx=ast.Load(),
            ),
            args=[ast.Constant(value=arg) for arg in args],
            keywords=[],
        )
        return ast.Expr(value=call)

    def _make_log_assignment(self, var_name: str, value_node: ast.expr) -> ast.Expr:
        """Create a log_assignment call"""
        self.injection_count += 1

        call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="__heisenberg_trace__", ctx=ast.Load()),
                attr="log_assignment",
                ctx=ast.Load(),
            ),
            args=[ast.Constant(value=var_name), value_node],
            keywords=[],
        )
        return ast.Expr(value=call)


class MinimalInstrumenter(BaseInstrumenter):
    """Minimal instrumentation: function calls and timing only"""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Add function call logging"""
        # Add call logging at start of function
        log_call = self._make_trace_call("log_call", node.name)

        # Insert at beginning of function body
        node.body.insert(0, log_call)

        # Continue visiting children
        self.generic_visit(node)
        return node


class StandardInstrumenter(BaseInstrumenter):
    """Standard instrumentation: function calls, assignments, loops"""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Add function call logging"""
        log_call = self._make_trace_call("log_call", node.name)
        node.body.insert(0, log_call)
        self.generic_visit(node)
        return node

    def visit_Assign(self, node: ast.Assign) -> list[ast.stmt]:
        """Log variable assignments"""
        result = [node]

        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                if not var_name.startswith("_"):
                    self.tracked_variables.append(var_name)
                    # Create a new Name node to reference the assigned value
                    value_ref = ast.Name(id=var_name, ctx=ast.Load())
                    log_stmt = self._make_log_assignment(var_name, value_ref)
                    result.append(log_stmt)

        return result

    def visit_For(self, node: ast.For) -> ast.For:
        """Add loop iteration counting"""
        loop_id = f"for_{self._loop_counter}"
        self._loop_counter += 1

        # Add iteration counter at start of loop body
        log_iter = self._make_trace_call("log_loop_iteration", loop_id)
        node.body.insert(0, log_iter)

        self.generic_visit(node)
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        """Add loop iteration counting"""
        loop_id = f"while_{self._loop_counter}"
        self._loop_counter += 1

        # Add iteration counter at start of loop body
        log_iter = self._make_trace_call("log_loop_iteration", loop_id)
        node.body.insert(0, log_iter)

        self.generic_visit(node)
        return node


class ComprehensiveInstrumenter(StandardInstrumenter):
    """Comprehensive instrumentation: everything plus intermediate values and branches"""

    def visit_If(self, node: ast.If) -> ast.If:
        """Log branch decisions"""
        branch_id = f"if_{self._branch_counter}"
        self._branch_counter += 1

        # Log branch taken at start of if body
        log_taken = self._make_trace_call("log_branch", branch_id, True)
        node.body.insert(0, log_taken)

        # Log branch not taken at start of else body (if exists)
        if node.orelse:
            log_not_taken = self._make_trace_call("log_branch", branch_id, False)
            if isinstance(node.orelse[0], ast.If):
                # elif - don't modify
                pass
            else:
                node.orelse.insert(0, log_not_taken)

        self.generic_visit(node)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        """Log intermediate calculation results (only in assignments)"""
        # This is complex to do correctly - would need parent context
        # For now, just pass through
        return node

    def visit_Return(self, node: ast.Return) -> list[ast.stmt]:
        """Log return values"""
        if node.value is not None:
            # Create a temp variable to hold return value
            temp_name = f"__return_val_{self.injection_count}__"

            # Assign to temp
            temp_assign = ast.Assign(
                targets=[ast.Name(id=temp_name, ctx=ast.Store())], value=node.value
            )

            # Return the temp
            new_return = ast.Return(value=ast.Name(id=temp_name, ctx=ast.Load()))

            return [
                temp_assign,
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="__heisenberg_trace__", ctx=ast.Load()),
                            attr="log_intermediate",
                            ctx=ast.Load(),
                        ),
                        args=[
                            ast.Constant(value="return_value"),
                            ast.Name(id=temp_name, ctx=ast.Load()),
                        ],
                        keywords=[],
                    )
                ),
                new_return,
            ]

        return [node]
