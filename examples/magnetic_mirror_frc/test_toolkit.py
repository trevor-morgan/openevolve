#!/usr/bin/env python3
"""
Test the DiscoveryToolkit with mock program data.

This tests external tool orchestration for true ontological discovery.
"""

import asyncio
from typing import Any

import numpy as np

from openevolve.discovery.golden_path import (
    Discovery,
    DiscoveryToolkit,
    ToolContext,
    create_default_toolkit,
)


def generate_programs(n: int = 50) -> list[dict[str, Any]]:
    """Generate mock programs with a hidden pattern."""
    np.random.seed(42)
    programs = []

    for i in range(n):
        # Hidden pattern: fitness ~ sqrt(n_coils) + 0.1 * log(code_length)
        n_coils = np.random.randint(3, 15)

        code_lines = ["def create_config():", "    coils = CoilSet()"]
        for j in range(n_coils):
            z = np.random.uniform(-2, 2)
            r = np.random.uniform(0.3, 1.5)
            curr = np.random.uniform(-1e6, 1e6)
            code_lines.append(f"    coils.add_coil(z={z:.2f}, radius={r:.2f}, current={curr:.0f})")
        code_lines.append("    return coils")
        code = "\n".join(code_lines)

        # Hidden relationship
        fitness = (
            0.3 + 0.1 * np.sqrt(n_coils) + 0.02 * np.log(len(code)) + np.random.normal(0, 0.03)
        )
        fitness = max(0.1, min(0.99, fitness))

        programs.append(
            {
                "iteration": i,
                "fitness": fitness,
                "metrics": {
                    "mirror_ratio": np.random.uniform(1, 20),
                    "well_depth": np.random.uniform(-0.5, 0.5),
                },
                "code": code,
                "program_id": f"prog_{i}",
            }
        )

    return programs


async def test_toolkit():
    """Test the discovery toolkit."""
    print("=" * 60)
    print("DISCOVERY TOOLKIT TEST")
    print("=" * 60)

    # Create toolkit
    toolkit = create_default_toolkit()

    available = toolkit.get_available_tools()
    print(f"\nAvailable tools: {len(available)}")
    for tool in available:
        print(f"  - {tool.name}: {tool.description[:50]}...")

    # Generate test data
    programs = generate_programs(100)
    print(f"\nGenerated {len(programs)} programs")
    print(
        f"Fitness range: {min(p['fitness'] for p in programs):.3f} - {max(p['fitness'] for p in programs):.3f}"
    )

    # Create context
    best = max(programs, key=lambda p: p["fitness"])
    context = ToolContext(
        programs=programs,
        current_metrics=["mirror_ratio", "well_depth"],
        domain_context="Magnetic mirror fusion optimization. Evolving coil configurations.",
        crisis_type="ontology_gap",
        best_fitness=best["fitness"],
        best_program_code=best["code"],
        discovery_goal="Find hidden variables that explain fitness variance",
    )

    # Run all tools
    print("\n" + "-" * 60)
    print("Running discovery tools...")
    print("-" * 60)

    discoveries = await toolkit.run_all_available(context)

    print(f"\nTotal discoveries: {len(discoveries)}")

    # Group by tool
    by_tool = {}
    for d in discoveries:
        by_tool.setdefault(d.source_tool, []).append(d)

    for tool, tool_discoveries in by_tool.items():
        print(f"\n{tool} ({len(tool_discoveries)} discoveries):")
        for d in tool_discoveries[:3]:
            print(f"  - {d.name}")
            print(f"    Type: {d.discovery_type.value}")
            print(f"    Confidence: {d.confidence:.2f}")
            if d.evidence:
                print(f"    Evidence: {d.evidence[0][:60]}...")
            if d.computation_code:
                print(f"    Has computation code: Yes ({len(d.computation_code)} chars)")

    # Test computation code execution
    print("\n" + "-" * 60)
    print("Testing computation code execution...")
    print("-" * 60)

    test_code = """
def create_config():
    coils = CoilSet()
    coils.add_coil(z=-1.5, radius=0.5, current=1000000)
    coils.add_coil(z=-0.5, radius=0.3, current=500000)
    coils.add_coil(z=0.5, radius=0.3, current=500000)
    coils.add_coil(z=1.5, radius=0.5, current=1000000)
    return coils
"""

    for d in discoveries:
        if d.computation_code:
            try:
                # Compile and execute
                namespace = {"np": np, "re": __import__("re"), "ast": __import__("ast")}
                exec(d.computation_code, namespace)

                # Find compute function
                compute_func = None
                for name, obj in namespace.items():
                    if callable(obj) and name.startswith("compute"):
                        compute_func = obj
                        break

                if compute_func:
                    value = compute_func(test_code, {"mirror_ratio": 5.0})
                    print(f"  {d.name}: {value}")
            except Exception as e:
                print(f"  {d.name}: FAILED - {e}")

    print("\n" + "=" * 60)
    print("TOOLKIT TEST COMPLETE")
    print("=" * 60)

    return discoveries


if __name__ == "__main__":
    asyncio.run(test_toolkit())
