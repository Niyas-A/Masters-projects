import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result
import open3d as o3d
from ICP import icp


# obj_name = 'drill' # drill or liq_container
# obj_name = 'drill' # drill or liq_container
obj_names = ['drill', 'liq_container']
n_samples = 4
reduction_ratio = 0.1  # For example, reduce to 10% of original size
for obj_name in obj_names:
    print('object: ',obj_name)
    source_pc = read_canonical_model(obj_name)
    num_points_source = len(source_pc)
    num_points_source_ds = int(num_points_source * reduction_ratio)
    voxel_size_source = np.sqrt(np.sum((source_pc.max(axis=0) - source_pc.min(axis=0)) ** 2) / num_points_source_ds)
    ds_pc = o3d.geometry.PointCloud()
    ds_pc.points = o3d.utility.Vector3dVector(source_pc)
    ds_pc = ds_pc.voxel_down_sample(voxel_size_source)
    ds_pc = np.asarray(ds_pc.points)
    for j in range(n_samples):
        target_pc = load_pc(obj_name, j)
        num_points_target = len(target_pc)
        num_points_target_ds = int(num_points_target * reduction_ratio)
        voxel_size_target = np.sqrt(np.sum((target_pc.max(axis=0) - target_pc.min(axis=0)) ** 2) / num_points_target_ds)
        dt_pc = o3d.geometry.PointCloud()
        dt_pc.points = o3d.utility.Vector3dVector(target_pc)
        dt_pc = dt_pc.voxel_down_sample(voxel_size_target)
        dt_pc = np.asarray(dt_pc.points)
        distance_values = []
        iter_values = []
        theta_values = np.pi*np.arange(0, 360, 30)/180
        dt_bar = np.average(target_pc, 0)
        ds_bar = np.average(source_pc, 0)
        T_init = np.eye(4)
        T_values = np.zeros((4,4,len(theta_values)), dtype=float)
        i=0
        for theta in theta_values:
            R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
            P = dt_bar - R@ds_bar
            T_init[:3, :3] = R
            T_init[:3, 3] = P
            T, distances, iter = icp(ds_pc,target_pc,T_init)
            distance_values.append(np.mean(distances))
            iter_values.append(iter)
            # print(theta, np.mean(distances), iter)
            T_values[:,:,i] = T
            i += 1
        print('mean distance for sample ',j,': ', min(distance_values),iter_values[np.argmin(distance_values)])
        Tf = T_values[:,:,np.argmin(distance_values)]
        # visualize_icp_result(source_pc,target_pc,Tf)
