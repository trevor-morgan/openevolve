"""
WebGL Export Utility.

Exports artifacts to a format suitable for Three.js visualization.
"""

import json
import os

import numpy as np


def export_to_webgl(artifacts: dict, output_path: str = "visualization.json"):
    """
    Export artifacts to JSON for WebGL.

    Handles numpy arrays and structures the data for standard loaders.
    """

    export_data = {"metadata": {"type": "openevolve_webgl", "version": "1.0"}, "scenes": []}

    # Check for field map
    if "field_map" in artifacts:
        field = artifacts["field_map"]
        # Convert to flat lists for buffers

        # Assume z_grid, r_grid are 1D arrays defining the mesh
        z = np.array(field.get("z_grid", []))
        r = np.array(field.get("r_grid", []))

        # Create full meshgrid coordinates
        Z, R = np.meshgrid(z, r)

        # Get B magnitude
        B = np.array(field.get("B_field", []))

        # Export as a surface
        # Vertices: [x, y, z] -> [Z, R, B] (mapping B to height) or just [Z, R, 0] with color=B

        vertices = []
        colors = []

        # Normalize B for color
        b_min, b_max = np.min(B), np.max(B)

        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                vertices.extend([float(Z[i, j]), float(R[i, j]), 0.0])  # Flat plane

                # Simple heatmap
                val = (B[i, j] - b_min) / (b_max - b_min + 1e-6)
                colors.extend([val, 0.0, 1.0 - val])  # Red=High, Blue=Low

        # Indices for triangles
        indices = []
        rows, cols = Z.shape
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Two triangles per quad
                # p1--p2
                # | / |
                # p3--p4
                p1 = i * cols + j
                p2 = i * cols + (j + 1)
                p3 = (i + 1) * cols + j
                p4 = (i + 1) * cols + (j + 1)

                indices.extend([p1, p3, p2])
                indices.extend([p2, p3, p4])

        export_data["scenes"].append(
            {
                "name": "Magnetic Field Magnitude",
                "type": "mesh",
                "vertices": vertices,
                "colors": colors,
                "indices": indices,
            }
        )

        # Check for particle trace

        if "particle_trace" in artifacts:
            # trace = artifacts["particle_trace"]

            # visualize z_final distribution as a 3D bar chart or point cloud?

            pass  # Todo

    with open(output_path, "w") as f:
        json.dump(export_data, f)

    # Generate HTML viewer
    html_path = output_path.replace(".json", ".html")
    with open(html_path, "w") as f:
        f.write(TEMPLATE.replace("DATA_SOURCE", os.path.basename(output_path)))


TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>OpenEvolve 3D Visualization</title>
    <style>body { margin: 0; }</style>
    <script type="importmap">
      {
        "imports": {
          "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
          "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
        }
      }
    </script>
</head>
<body>
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        camera.position.z = 5;

        fetch('DATA_SOURCE')
            .then(response => response.json())
            .then(data => {
                data.scenes.forEach(item => {
                    if (item.type === 'mesh') {
                        const geometry = new THREE.BufferGeometry();
                        geometry.setAttribute('position', new THREE.Float32BufferAttribute(item.vertices, 3));
                        geometry.setAttribute('color', new THREE.Float32BufferAttribute(item.colors, 3));
                        geometry.setIndex(item.indices);

                        const material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, wireframe: false });
                        const mesh = new THREE.Mesh(geometry, material);
                        scene.add(mesh);
                    }
                });
            });

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>
"""
