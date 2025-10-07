#!/usr/bin/env python3
"""
Spatially-Aware GLB to Bricks converter - CORRECTED VERSION.
This version directly maps 3D positions to vertex colors without complex UV interpolation.
"""

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from pathlib import Path
import argparse
from typing import Tuple, Dict, Any, List
from scipy.spatial import cKDTree


def extract_vertex_colors_from_texture(mesh: trimesh.Trimesh) -> np.ndarray:
    """Extract colors at each vertex using UV coordinates."""
    if not hasattr(mesh.visual, 'material') or mesh.visual.material is None:
        return None
    
    material = mesh.visual.material
    if not hasattr(material, 'baseColorTexture') or material.baseColorTexture is None:
        return None
    
    if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
        return None
    
    texture_image = material.baseColorTexture
    if texture_image.mode != 'RGB':
        texture_image = texture_image.convert('RGB')
    
    texture_array = np.array(texture_image)
    height, width = texture_array.shape[:2]
    uv_coords = mesh.visual.uv
    
    # Sample colors at each vertex using its UV coordinate
    vertex_colors = np.zeros((len(mesh.vertices), 3))
    for i, uv in enumerate(uv_coords):
        u_pixel = int(np.clip(uv[0], 0, 1) * (width - 1))
        v_pixel = int(np.clip((1 - uv[1]), 0, 1) * (height - 1))  # Flip V
        vertex_colors[i] = texture_array[v_pixel, u_pixel, :3] / 255.0
    
    return vertex_colors


def sample_color_at_position(position: np.ndarray, 
                           mesh_vertices: np.ndarray, 
                           vertex_colors: np.ndarray,
                           kdtree: cKDTree,
                           k_neighbors: int = 5) -> np.ndarray:
    """
    Sample color at a 3D position using k-nearest neighbor interpolation of vertex colors.
    
    This approach directly uses the vertex colors we extracted, avoiding UV complexity.
    """
    # Find k nearest vertices to this position
    distances, indices = kdtree.query(position, k=k_neighbors)
    
    # Avoid division by zero
    distances = np.maximum(distances, 1e-8)
    
    # Use inverse distance weighting
    weights = 1.0 / distances
    weights = weights / np.sum(weights)  # Normalize
    
    # Interpolate colors using weighted average
    interpolated_color = np.zeros(3)
    for i, (weight, vertex_idx) in enumerate(zip(weights, indices)):
        interpolated_color += weight * vertex_colors[vertex_idx]
    
    return np.clip(interpolated_color, 0, 1)


def create_spatially_accurate_bricks(glb_path: str,
                                   world_size: float = 25.0,
                                   voxel_size: float = 0.8,
                                   k_neighbors: int = 5) -> Tuple[o3d.geometry.TriangleMesh, Dict]:
    """Create bricks with spatially accurate color mapping."""
    print(f"Creating spatially accurate bricks from: {glb_path}")
    
    # Load GLB
    scene = trimesh.load(glb_path)
    if not isinstance(scene, trimesh.Scene) or len(scene.geometry) == 0:
        raise ValueError("No geometries found in GLB file")
    
    geometry_name, mesh = next(iter(scene.geometry.items()))
    print(f"Processing geometry '{geometry_name}': {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Extract vertex colors
    vertex_colors = extract_vertex_colors_from_texture(mesh)
    color_source = "none"
    
    if vertex_colors is not None:
        color_source = "texture"
        print(f"  Extracted colors from texture")
        print(f"  Color range: [{vertex_colors.min():.3f}, {vertex_colors.max():.3f}]")
        
        # Analyze color distribution
        unique_colors = len(np.unique(vertex_colors.reshape(-1, 3), axis=0))
        print(f"  Unique vertex colors: {unique_colors}")
        
    elif hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
        material = mesh.visual.material
        if hasattr(material, 'baseColorFactor') and material.baseColorFactor is not None:
            base_color = material.baseColorFactor[:3] / 255.0
            vertex_colors = np.tile(base_color, (len(mesh.vertices), 1))
            color_source = "material"
            print(f"  Using base color factor: {material.baseColorFactor[:3]}")
    
    if vertex_colors is None:
        # Default colors based on Y position
        y_coords = mesh.vertices[:, 1]
        y_min, y_max = y_coords.min(), y_coords.max()
        y_threshold = y_min + (y_max - y_min) * 0.4
        
        vertex_colors = np.zeros((len(mesh.vertices), 3))
        for i, y in enumerate(y_coords):
            if y <= y_threshold:
                vertex_colors[i] = [0.4, 0.2, 0.1]  # Brown
            else:
                vertex_colors[i] = [0.2, 0.6, 0.2]  # Green
        color_source = "default"
        print(f"  Using default position-based colors")
    
    print(f"  Color source: {color_source}")
    
    # Store original data before transformation
    original_vertices = mesh.vertices.copy()
    original_vertex_colors = vertex_colors.copy()
    
    # Convert to Open3D for voxelization
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    
    # Normalize mesh while preserving proportions
    center = o3d_mesh.get_center()
    o3d_mesh.translate(-center)
    bbox = o3d_mesh.get_max_bound() - o3d_mesh.get_min_bound()
    scale_factor = world_size / np.max(bbox)
    o3d_mesh.scale(scale_factor, center=np.array([0, 0, 0]))
    
    print(f"  Scale factor: {scale_factor:.4f}")
    print(f"  New extents: {o3d_mesh.get_axis_aligned_bounding_box().get_extent()}")
    
    # Transform original vertices to match
    transformed_vertices = (original_vertices - center) * scale_factor
    
    # Create KDTree for fast nearest neighbor search
    print(f"  Building KDTree for {len(transformed_vertices)} vertices...")
    kdtree = cKDTree(transformed_vertices)
    
    # Create voxel grid
    print(f"  Creating voxel grid with voxel size: {voxel_size}")
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(o3d_mesh, voxel_size)
    voxels = voxel_grid.get_voxels()
    
    if len(voxels) == 0:
        raise ValueError("No voxels generated!")
    
    print(f"  Generated {len(voxels)} voxels")
    
    # CRITICAL: Get the voxel grid origin to align coordinates properly
    voxel_grid_origin = voxel_grid.origin
    print(f"  Voxel grid origin: {voxel_grid_origin}")
    
    # Test color sampling at a few positions
    test_positions = [
        np.array([0, 0, 0]),        # Center
        np.array([0, -10, 0]),      # Bottom
        np.array([0, 10, 0]),       # Top
    ]
    
    # Test with actual voxel grid coordinates
    test_positions = [
        voxel_grid_origin + np.array([0, 0, 0]),                    # Grid origin
        voxel_grid_origin + np.array([0, voxel_size * 5, 0]),      # A bit up
        voxel_grid_origin + np.array([0, voxel_size * 25, 0]),     # Near top
    ]
    
    print("  Testing color sampling with voxel grid coordinates:")
    for i, pos in enumerate(test_positions):
        color = sample_color_at_position(pos, transformed_vertices, original_vertex_colors, kdtree, k_neighbors)
        pos_name = ["Grid Origin", "Lower", "Higher"][i]
        print(f"    {pos_name} {pos}: {color}")
    
    # Create colored bricks
    all_bricks = []
    color_samples = []
    
    print(f"  Creating {len(voxels)} colored bricks...")
    for i, voxel in enumerate(voxels):
        if i % 500 == 0:
            print(f"    Processing voxel {i+1}/{len(voxels)}")
        
        grid_idx = voxel.grid_index
        
        # Get voxel center in world coordinates (accounting for voxel grid origin)
        voxel_center = np.array([
            voxel_grid_origin[0] + grid_idx[0] * voxel_size,
            voxel_grid_origin[1] + grid_idx[1] * voxel_size,
            voxel_grid_origin[2] + grid_idx[2] * voxel_size
        ])
        
        # Sample color at this position
        voxel_color = sample_color_at_position(
            voxel_center, 
            transformed_vertices, 
            original_vertex_colors, 
            kdtree, 
            k_neighbors
        )
        
        color_samples.append(voxel_color)
        
        # Create brick at the correct position
        brick = o3d.geometry.TriangleMesh.create_box(
            voxel_size * 0.95, voxel_size * 0.95, voxel_size * 0.95
        )
        brick.translate(voxel_center)
        brick.paint_uniform_color(voxel_color)
        
        all_bricks.append(brick)
    
    # Analyze final results
    color_samples = np.array(color_samples)
    unique_colors = len(np.unique(color_samples.reshape(-1, 3), axis=0))
    
    print(f"  Final analysis:")
    print(f"    Unique brick colors: {unique_colors}")
    print(f"    Color range: [{color_samples.min():.3f}, {color_samples.max():.3f}]")
    print(f"    Mean color: {color_samples.mean(axis=0)}")
    print(f"    Color std: {color_samples.std(axis=0)}")
    
    # Combine all bricks
    print(f"  Combining {len(all_bricks)} bricks...")
    combined_mesh = all_bricks[0]
    for brick in all_bricks[1:]:
        combined_mesh += brick
    
    combined_mesh.compute_vertex_normals()
    
    # Create info
    info = {
        'total_bricks': len(all_bricks),
        'unique_colors': unique_colors,
        'color_source': color_source,
        'world_size': world_size,
        'voxel_size': voxel_size,
        'scale_factor': scale_factor,
        'k_neighbors': k_neighbors,
        'method': 'k-nearest-neighbor-interpolation'
    }
    
    return combined_mesh, info


def main():
    parser = argparse.ArgumentParser(description="Spatially Accurate GLB to Bricks Converter")
    parser.add_argument("input", nargs="?", default="tree2.glb", help="Input GLB file")
    parser.add_argument("--world-size", type=float, default=25.0, help="World size")
    parser.add_argument("--voxel-size", type=float, default=0.8, help="Voxel size")
    parser.add_argument("--k-neighbors", type=int, default=5, help="Number of nearest neighbors for interpolation")
    parser.add_argument("--output", help="Output PLY file")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: File '{args.input}' not found!")
        return 1
    
    if not args.output:
        input_stem = Path(args.input).stem
        args.output = f"{input_stem}_spatially_accurate_bricks.ply"
    
    try:
        # Create spatially accurate bricks
        brick_mesh, info = create_spatially_accurate_bricks(
            args.input,
            world_size=args.world_size,
            voxel_size=args.voxel_size,
            k_neighbors=args.k_neighbors
        )
        
        # Save results
        print(f"\nSaving to: {args.output}")
        o3d.io.write_triangle_mesh(args.output, brick_mesh)
        
        # Save info
        info_path = args.output.replace('.ply', '_info.txt')
        with open(info_path, 'w') as f:
            f.write("Spatially Accurate GLB to Bricks Conversion\n")
            f.write("=" * 45 + "\n")
            for key, value in info.items():
                f.write(f"{key}: {value}\n")
        
        # Visualize if requested
        if args.visualize:
            print("\nShowing spatially accurate brick structure...")
            o3d.visualization.draw_geometries(
                [brick_mesh], 
                window_name="Spatially Accurate Brick Structure",
                width=1200, height=900
            )
        
        print(f"\n✅ Success!")
        print(f"Created {info['total_bricks']} spatially accurate bricks")
        print(f"Unique colors: {info['unique_colors']}")
        print(f"Color source: {info['color_source']}")
        print(f"Method: {info['method']}")
        print(f"Output: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())