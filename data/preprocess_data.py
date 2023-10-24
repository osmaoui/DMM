import json
from mesh_to_sdf import sample_sdf_near_surface
import trimesh
import pyrender
import numpy as np
import polyscope as ps
from glob import glob
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from colormap import hex2rgb
import pickle


fdi_colors = {
               "18": "#ffe2e2", "17": "#ffc6c6", "16" : "#ffaaaa", "15":"#ff8d8d", "14":"#ff7171", "13":"#ff5555",
    "12": "#ff3838", "11": "#ff0000", "21": "#0000ff", "22": "#3838ff", "23": "#5555ff", "24": "#7171ff", "25":"#8d8dff","26":"#aaaaff",
    "27": "#c6c6ff",
    "28": "#e2e2ff",
    "38":"#001c00",
     "37":"#003800",
    "36":"#005500",
     "35":"#007100",
     "34":"#008d00",
     "33":"#00aa00",
     "32":"#00c600",
     "31":"#00ff00",
     "48":"#8000ff",
     "47":"#9c38ff",
     "46":"#aa55ff",
     "45":"#b871ff",
     "44":"#c68dff",
     "43":"#d4aaff",
     "42":"#e2c6ff",
     "41":"#f0e2ff",
     "0":"#000000"
}
def load_template(template_dir_path):
    template_paths = glob(template_dir_path + '/meanshape*')
    mesh_dict = dict()
    for template_path in template_paths:
        teeth_id = template_path.split('/')[-1].split('_')[-1].split('.ply')[0]
        if teeth_id == '00':
            continue
        mesh_dict[teeth_id] = np.mean(trimesh.load(template_path, process=False).vertices, axis=0)
    return mesh_dict


def load_scan(scan_path):
    mesh = trimesh.load(scan_path, process=False)
    with open(scan_path.replace('.obj', '.json'), 'r') as f:
        gt_label_dict = json.load(f)

    gt_instances = np.array(gt_label_dict['instances'])
    gt_labels = np.array(gt_label_dict['labels'])

    u_instances = np.unique(gt_instances)
    u_instances = u_instances[u_instances != 0]

    gt_instance_label_dict = {}
    for l in u_instances:
        gt_lbl = gt_labels[gt_instances == l]
        label = np.unique(gt_lbl)

        assert len(label) == 1
        # compute gt tooth center and size
        gt_verts = mesh.vertices[gt_instances == l]
        gt_center = np.mean(gt_verts, axis=0)
        gt_instance_label_dict[str(label[0])] = gt_center

    return gt_instance_label_dict, mesh, gt_labels


if __name__ == "__main__":
    teeth = np.load(
        '/media/oussama/60d0458f-2f1f-4c73-bfe4-93757a0b94c52/home/oussama/Downloads/test_data/teeth_0000.npz'
    )
    # ps.init()
    # ps.register_point_cloud('center', np.array([[0,0,0]]))
    # ps.register_point_cloud('pos', teeth['pos'][:, :3])
    # ps.register_point_cloud('neg', teeth['neg'][:, :3])
    # ps.register_point_cloud('surf', teeth['surf'][:, :3])
    # ps.show()
    scan_dir_path = '/media/oussama/60d0458f-2f1f-4c73-bfe4-93757a0b94c52/3D_Data/Teeth3DS/Teeth3DS/training/upper'
    template_centroids = load_template('/media/oussama/60d0458f-2f1f-4c73-bfe4-93757a0b94c52/home/oussama/workspace/DMM/examples/upper_dmm/Reconstructions/240/Meshes')
    centroids = np.zeros((50, 3))
    for lbl, cent in template_centroids.items():
        centroids[int(lbl)] = cent
    np.savetxt('../test_data/avg_centroids.txt', centroids)
    scans_path = glob(scan_dir_path + '/*/*upper.obj')
    for nb, scan_path in enumerate(tqdm(scans_path)):
        patient_id = scan_path.split('/')[-1].split('.obj')[0]
        if nb > 9:
            break
        scan_centroids, candidate_mesh, gt_labels = load_scan(scan_path)
        labels_list = [0]
        points_set1, points_set2 = [], []
        for label, cent in scan_centroids.items():
            points_set1.append(cent)
            points_set2.append(template_centroids[label])
            labels_list.append(int(label))
        # translate all the data to the origin
        cent_scan = np.mean(points_set1, 0)
        points_set1 -= cent_scan
        cent_template = np.mean(points_set2, 0)
        points_set2 -= cent_template
        norm1 = np.linalg.norm(points_set1)
        norm2 = np.linalg.norm(points_set2)

        points_set1 /= norm1
        points_set2 /= norm2

        R, s = orthogonal_procrustes(points_set1, points_set2)
        mtx2 = np.dot(points_set1, R.T) * s

        candidate_mesh.vertices = (np.dot((candidate_mesh.vertices - cent_scan)/norm1, R.T) * s) * norm2 + cent_template

        # ps.init()
        # ps.register_point_cloud('mesh', candidate_mesh.vertices)

        for lbl, cent in scan_centroids.items():
            scan_centroids[lbl] = (np.dot((cent - cent_scan) / norm1, R.T) * s) * norm2 + cent_template
        #     ps.register_point_cloud(lbl, scan_centroids[lbl].reshape(1, 3))
        # ps.show()

        candidate_mesh.export('../test_data/' + patient_id + '.obj')
        epsilon = 1e-3
        (points, sdf), pcd = sample_sdf_near_surface(candidate_mesh, number_of_points=250000, surface_point_method='sample', sign_method='normal', return_gradients=False, sample_point_count=1000000)

        knn_model_src = NearestNeighbors(n_neighbors=1)
        knn_model_src.fit(np.asarray(candidate_mesh.vertices))
        distances_src, indices_src = knn_model_src.kneighbors(pcd.points)
        color, labels = [], []
        for i in indices_src:
            color.append(list(hex2rgb(fdi_colors[str(gt_labels[i[0]])])))
            labels.append(gt_labels[i[0]])
        # ps.init()
        # ps_mesh = ps.register_point_cloud('teeth', pcd.points)
        # ps_mesh.add_color_quantity('colors', np.array(color) / 255)
        # ps.show()

        with open('../test_data/SdfSamples/' + patient_id + '.pkl', 'wb') as pickle_file:
            pickle.dump(scan_centroids, pickle_file)
        np.savez('../test_data/SdfSamples/' + patient_id + '.npz', surf=np.hstack((pcd.points, np.array([0] * len(pcd.points)).reshape(-1, 1), np.array(labels).reshape(-1, 1))).astype(np.float32),
                 pos=np.hstack(
                     (points[sdf > 0], sdf[sdf > 0].reshape(-1, 1), np.array([-1] * len(points[sdf > 0])).reshape(-1, 1))).astype(np.float32),
                 neg=np.hstack(
                     (points[sdf < 0], sdf[sdf < 0].reshape(-1, 1), np.array([-1] * len(points[sdf < 0])).reshape(-1, 1))).astype(np.float32),
                 normals=pcd.normals.astype(np.float32))
        np.savetxt('../test_data/' + patient_id + '.txt',
                   np.array(labels_list).astype(int).reshape(len(labels_list), 1), fmt='%d')
        colors = np.zeros(points.shape)
        colors[sdf < 0, 2] = 1
        colors[sdf > 0, 0] = 1
        colors[sdf == 0] = [0.5, 0.5, 0.5]
        ps.init()
        ps.register_point_cloud('pos', points[sdf > epsilon])
        ps.register_point_cloud('neg', points[sdf < -epsilon])
        ps.register_point_cloud('surf', points[(sdf < epsilon) & (sdf > -epsilon)])
        ps.register_point_cloud('pcd', pcd.points)
        ps.register_point_cloud('vertices', candidate_mesh.vertices)
        ps.register_point_cloud('surf_refrence', teeth['surf'][:, :3])
        ps.register_point_cloud('center', np.array([[0,0,0]]))
        ps.show()
        # cloud = pyrender.Mesh.from_points(points, colors=colors)
        # scene = pyrender.Scene()
        # scene.add(cloud)
        # viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
