import open3d as o3d
import numpy as np
import trimesh

teeth = np.load(
    '/media/oussama/60d0458f-2f1f-4c73-bfe4-93757a0b94c52/home/oussama/Downloads/test_data/teeth_0000.npz'
)
sampled_indices = np.random.choice(len(teeth['surf']), size=200000, replace=False)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(teeth['surf'][:,:3][sampled_indices])
pcd.normals = o3d.utility.Vector3dVector(teeth['normals'][sampled_indices])
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 1.5 * avg_dist

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
           pcd,
           o3d.utility.DoubleVector([radius, radius * 2]))

# create the triangular mesh with the vertices and faces from open3d
tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                          vertex_normals=np.asarray(mesh.vertex_normals))

trimesh.convex.is_convex(tri_mesh)
tri_mesh.export('../test_data/teeth_0000.obj')