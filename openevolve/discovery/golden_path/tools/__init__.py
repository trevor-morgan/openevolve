"""
Discovery Tools - External frameworks for ontological discovery.

Each tool wraps an external framework/API and produces Discovery objects
that can be validated and integrated by the Golden Path.

Available tools:
- SymbolicRegressionTool: PySR/gplearn for discovering mathematical formulas
- CausalDiscoveryTool: DoWhy/causal-learn for causal structure
- WebResearchTool: arXiv, Semantic Scholar, web search
- CodeAnalysisTool: AST-based structural pattern analysis
- WolframTool: Wolfram Alpha for analytical solutions
"""

# Tools are imported lazily by the toolkit to handle missing dependencies
