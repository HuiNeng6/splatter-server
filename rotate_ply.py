import numpy as np
from plyfile import PlyData, PlyElement
# Use scipy for robust quaternion handling
from scipy.spatial.transform import Rotation as R

def apply_4x4_transformation_scipy(input_path, output_path, transform_matrix):
    """
    Applies a 4x4 transformation matrix to Gaussian Splats (position, rotation, scale)
    in a PLY file using SciPy.
    """
    plydata = PlyData.read(input_path)
    vertices = plydata['vertex'].data
    
    R_world_matrix = transform_matrix[:3, :3]
    
     # --- Transform Rotation (position) ---

    positions = np.vstack((vertices['x'], vertices['y'], vertices['z'])).T
    # Apply transformation to positions using matrix multiplication
    # P_hom = [P | 1], P'_hom = T @ P_hom, P' = P'_hom[:3] / P'_hom[3]
    positions_hom = np.hstack((positions, np.ones((positions.shape[0], 1))))
    transformed_positions_hom = (transform_matrix @ positions_hom.T).T
    transformed_positions = transformed_positions_hom[:, :3] / transformed_positions_hom[:, 3, np.newaxis]

    vertices['x'] = transformed_positions[:, 0]
    vertices['y'] = transformed_positions[:, 1]
    vertices['z'] = transformed_positions[:, 2]
    
    # --- Transform Rotation (quaternions) ---

    # 1. Extract original quaternions (typically WXYZ in the PLY file)
    quats_original_wxyz = np.stack([
        vertices['rot_0'], # W
        vertices['rot_1'], # X
        vertices['rot_2'], # Y
        vertices['rot_3']  # Z
    ], axis=1)

    # 2. Convert WXYZ format to SciPy's default XYZW (scalar-last) format
    quats_original_xyzw = quats_original_wxyz[:, [1, 2, 3, 0]]
    
    # 3. Create a SciPy Rotation object from the original quaternions
    # SciPy normalizes the quaternions automatically on creation.
    rotations_original = R.from_quat(quats_original_xyzw)

    # 4. Create a SciPy Rotation object for the transformation matrix
    # This automatically converts the 3x3 rotation matrix component to a quaternion internally
    rotation_transform = R.from_matrix(R_world_matrix)
    
    # 5. Apply the rotation transformation (composition using the * operator)
    # The result is a new set of Rotation objects representing the combined rotation
    # print(rotation_transform.shape)
    # print(rotations_original.as_matrix().shape)

    # transformed_rotations = rotations_original * rotation_transform  
    transformed_rotations = rotation_transform * rotations_original
    # transformed_rotations_np = R_world_matrix.T @ rotations_original.as_matrix() @ R_world_matrix
    # transformed_rotations = R.from_matrix(transformed_rotations_np)

    # 6. Convert back to XYZW float array and reorder to WXYZ for saving to PLY
    transformed_quats_xyzw = transformed_rotations.as_quat()
    transformed_quats_wxyz = transformed_quats_xyzw[:, [3, 0, 1, 2]]

    # 7. Assign back to the structured array fields
    vertices['rot_0'] = transformed_quats_wxyz[:, 0] # W
    vertices['rot_1'] = transformed_quats_wxyz[:, 1] # X
    vertices['rot_2'] = transformed_quats_wxyz[:, 2] # Y
    vertices['rot_3'] = transformed_quats_wxyz[:, 3] # Z

    # --- Transform Scale (scales remain unchanged for rigid/uniform transformations) ---

    # 8. Create a new PlyElement and save the file
    el = PlyElement.describe(vertices, 'vertex')
    new_plydata = PlyData([el], text=plydata.text)
    new_plydata.write(output_path)
    
    print(f"Transformed file saved to {output_path}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Copy images and extra files to an output folder")
    parser.add_argument("--input", required=True, help="Input ply file path")
    parser.add_argument("--output", required=True, help="Output ply file path")
    args = parser.parse_args()

    trans3 = np.zeros((4,4))
    trans3[0,0] = 1
    trans3[1,1] = -1
    trans3[2,2] = -1
    trans3[3,3] = 1

    trans = np.zeros((4,4))
    trans[0,2] = -1
    trans[1,0] = 1
    trans[2,1] = -1
    trans[3,3] = 1
    print(trans)
    trans = trans3 @ trans
    print(trans)
    # Apply the rotation matrix
    apply_4x4_transformation_scipy(args.input, args.output, trans)
