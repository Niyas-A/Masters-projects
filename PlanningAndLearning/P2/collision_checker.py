import numpy as np
import fcl
from scipy.spatial.transform import Rotation as R


def check_collision(path, blocks):
    collision = False
    for i in range(len(path) - 1):
        segment_start = np.array(path[i])
        segment_end = np.array(path[i + 1])
        direction = segment_end - segment_start
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length
        else:
            direction = np.zeros_like(direction)

        # Create a capsule for the path segment
        capsule = fcl.Capsule(radius=0.001, lz=length)
        T = (segment_start + segment_end) * 0.5
        z_axis = np.array([0, 0, 1])

        if np.allclose(direction, z_axis):
            q = R.from_quat([0, 0, 0, 1]).as_quat()  # Identity quaternion
        else:
            # Calculate the axis of rotation (cross product of z-axis and direction)
            rotation_axis = np.cross(z_axis, direction)
            rotation_axis_norm = np.linalg.norm(rotation_axis)
            if rotation_axis_norm != 0:
                rotation_axis = rotation_axis / rotation_axis_norm

            # Calculate the angle of rotation (dot product of z-axis and direction)
            dot_product = np.dot(z_axis, direction)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
            # Create the quaternion from the rotation axis and angle
            q = R.from_rotvec(rotation_axis * angle).as_quat()

        tf_capsule = fcl.Transform(q, T)
        O1 = fcl.CollisionObject(capsule, tf_capsule)

        for block in blocks:
            block_min = np.array(block[0:3])
            block_max = np.array(block[3:6])
            box_dims = block_max - block_min

            # Create a box for the AABB
            aabb = fcl.Box(*box_dims)
            T_aabb = block_min + box_dims * 0.5
            tf_aabb = fcl.Transform(T_aabb)

            O2 = fcl.CollisionObject(aabb, tf_aabb)

            # Check for collision
            request = fcl.CollisionRequest()
            result = fcl.CollisionResult()
            ret = fcl.collide(O1, O2, request, result)
            # print(ret)
            if ret > 0:
                # print('collision with segments: ',segment_start, segment_start )
                collision = True
                break
        if collision:
            break
    return collision