
import pdb

import sys
import numpy as np
from pathlib import Path
import seaborn as sns
import argparse
from multiprocessing import Pool

from visualization.vis_tools import load_run_output_files, get_controller_data,\
    init_env, load_all_inds, load_all_behavior_descriptors, get_folder_path_from_str, load_all_fitnesses
from algorithms.evaluate import init_controller, stabilize_simulation
from algorithms.archives import structured_archive as sa


N_ITER_STABILIZING_SIM_DISPLAY = 1000

LINE_WIDTH_DOWNSCALE_FACTOR = 30
LINE_WIDTH_SCALE_FACTOR = 30
LINE_WIDTH_TOP_N_SCALE_FACTOR = 100

DEBUG = False

CONTACT_OBJ_DIMS = 0.01, 0.01, 0.01
SHADOW_OBJ_DIMS = 0.01, 0.01, 0.01
SHADOW_OBJ_COLOR = [1, 0, 0, 1]
CONTACT_OBJ_COLOR = [0, 1, 0, 1]

IDX_CONTACT_PT_POS_ON_OBJ = 5

max_x_touch = sa.MAX_X_TOUCH_VAL
max_y_touch = sa.MAX_Y_TOUCH_VAL
max_z_touch = sa.MAX_Z_TOUCH_VAL

min_x_touch = sa.MIN_X_TOUCH_VAL
min_y_touch = sa.MIN_Y_TOUCH_VAL
min_z_touch = sa.MIN_Z_TOUCH_VAL

n_bins_x_touch = sa.N_BINS_PER_DIM_POS_X_TOUCH
n_bins_y_touch = sa.N_BINS_PER_DIM_POS_Y_TOUCH
n_bins_z_touch = sa.N_BINS_PER_DIM_POS_Z_TOUCH

len_bin_x_touch = (max_x_touch - min_x_touch) / n_bins_x_touch
len_bin_y_touch = (max_y_touch - min_y_touch) / n_bins_y_touch
len_bin_z_touch = (max_z_touch - min_z_touch) / n_bins_z_touch


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runs", help="The directory containing runs", type=str,
                        default=str(Path(__file__).parent.parent / 'runs'))
    return parser.parse_args()


# Global variables for parallelization
# todo refactore: scoop is not used anymore - global var should be avoided now

args = arg_parser()
run_folder_path = args.runs
folder = get_folder_path_from_str(run_folder_path)
run_details, run_infos, cfg = load_run_output_files(folder)

CONTROLLER_CLASS, CONTROLLER_INFO = get_controller_data(cfg)
NB_ITER = CONTROLLER_INFO['nb_iter']
CONTROLLER_INFO['env_name'] = cfg['robot']['kwargs']['env_name']

ENV = init_env(cfg, display=False)

INDIVIDUALS = load_all_inds(folder)
BEHAVIOR_DESCRIPTORS = load_all_behavior_descriptors(folder)
FITNESSES = load_all_fitnesses(folder)

if DEBUG:
    N_MAX_INDS_DEBUG = 100
    INDIVIDUALS = INDIVIDUALS[:N_MAX_INDS_DEBUG]
    BEHAVIOR_DESCRIPTORS = BEHAVIOR_DESCRIPTORS[:N_MAX_INDS_DEBUG]
    FITNESSES = FITNESSES[:N_MAX_INDS_DEBUG]


#  Display parameters

# ----- Arrow

arrow_side = len_bin_x_touch / 2

# ----- Contact point object

co_dim_x, co_dim_y, co_dim_z = len_bin_x_touch, len_bin_y_touch, len_bin_z_touch
display_scale_factor = 4
info_shape_contact = {"shapeType": ENV.p.GEOM_BOX,
                      "halfExtents": [
                          co_dim_x / display_scale_factor,
                          co_dim_y / display_scale_factor,
                          co_dim_z / display_scale_factor
                      ]}


def extract_end_effector_orientation_at_touch(ind):

    ENV.reset(load='state', force_state_load=True)
    stabilize_simulation(env=ENV)
    nrmlized_pos_arm = ENV.get_joint_state(normalized=True)

    controller = init_controller(
        individual=ind,
        controller_class=CONTROLLER_CLASS,
        controller_info=CONTROLLER_INFO,
        env=ENV
    )

    end_eff_or_euler = None
    for i_step in range(NB_ITER):

        action = controller.get_action(i_step=i_step, nrmlized_pos_arm=nrmlized_pos_arm, env=ENV)

        _, reward, _, info = ENV.step(action)

        nrmlized_pos_arm = ENV.get_joint_state(normalized=True)

        if info['touch']:
            end_eff_or_quaterions = ENV.p.getLinkState(ENV.robot_id, ENV.end_effector_id, computeLinkVelocity=True)[1]
            end_eff_or_euler = ENV.p.getEulerFromQuaternion(end_eff_or_quaterions)
            break

    return end_eff_or_euler


def extract_trajectories(individuals):
    n_inds = len(individuals)
    print(f'Extracting orientations associated to each of the {n_inds} individuals from the successful archive ...',
          end='')

    with Pool() as p:
        all_end_eff_or_euler = np.array(p.map(extract_end_effector_orientation_at_touch, individuals))

    print('done.')
    print(f'{len(all_end_eff_or_euler)} obtained orientations.')

    i_inds2skip = [
        i_ind for i_ind, end_eff_or_euler in enumerate(all_end_eff_or_euler) if end_eff_or_euler is None
    ]

    print(f'Note: Some trajs might be discarded due to simulation non-determinism during extraction. (This should only '
          f'discard very fragile trajectories).'
          f'Number of discarded trajs : {len(i_inds2skip)}')

    return all_end_eff_or_euler, i_inds2skip


def get_i_cell_dims(ind_bds):
    target_min_x, target_min_y, target_min_z = min_x_touch, min_y_touch, min_z_touch
    target_len_bin_x, target_len_bin_y, target_len_bin_z = \
        len_bin_x_touch, len_bin_y_touch, len_bin_z_touch
    target_n_bin_x, target_n_bin_y, target_n_bin_z = \
        n_bins_x_touch, n_bins_y_touch, n_bins_z_touch

    pos_x, pos_y, pos_z = ind_bds

    i_bin_x = np.floor((pos_x - target_min_x) / target_len_bin_x)
    i_bin_y = np.floor((pos_y - target_min_y) / target_len_bin_y)
    i_bin_z = np.floor((pos_z - target_min_z) / target_len_bin_z)

    # corner cases in which the value is above the specified max -> set to last bin
    i_bin_x = target_n_bin_x - 1 if i_bin_x > target_n_bin_x else i_bin_x
    i_bin_y = target_n_bin_y - 1 if i_bin_y > target_n_bin_y else i_bin_y
    i_bin_z = target_n_bin_z - 1 if i_bin_z > target_n_bin_z else i_bin_z

    pt_x_center = min_x_touch + len_bin_x_touch / 2 + i_bin_x * len_bin_x_touch
    pt_y_center = min_y_touch + len_bin_y_touch / 2 + i_bin_y * len_bin_y_touch
    pt_z_center = min_z_touch + len_bin_z_touch / 2 + i_bin_z * len_bin_z_touch

    return pt_x_center, pt_y_center, pt_z_center


def get_map_cells(inds_bds):

    success_archive_cell_poses = []

    for ind_bds in inds_bds:

        pt_x_center, pt_y_center, pt_z_center = get_i_cell_dims(ind_bds)
        cell_center_coord = (pt_x_center, pt_y_center, pt_z_center)

        assert cell_center_coord not in success_archive_cell_poses, \
            'error in cell compute : 2 inds found for a single cell'

        success_archive_cell_poses.append(cell_center_coord)

    return success_archive_cell_poses


def generate_obj_id(env, info_shape_contact, color):
    base_collision_shape = env.p.createCollisionShape(**info_shape_contact)
    base_vis_shape = env.p.createVisualShape(**info_shape_contact, rgbaColor=color)
    return env.p.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=base_collision_shape,
                                 baseVisualShapeIndex=base_vis_shape,
                                 useMaximalCoordinates=False)


def cos(x):
    return np.cos(x)


def sin(x):
    return np.sin(x)


def cvt_fit2linewidth(fit, max_fit, min_fit):
    # fit -> [0, 1]
    return (max_fit - min_fit) * fit * LINE_WIDTH_SCALE_FACTOR


def get_arrow_tip_pose(contact_pos, end_eff_or_euler):
    theta_x, theta_y, theta_z = end_eff_or_euler

    rot_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0, 0],
        [np.sin(theta_z), np.cos(theta_z), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    rot_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1],
    ])
    rot_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x), 0],
        [0, np.sin(theta_x), np.cos(theta_x), 0],
        [0, 0, 0, 1],
    ])

    contact_center_arrow_tip_pose = np.array([0, 0, arrow_side, 0])
    contact_center_arrow_tip_pose_right_side = np.array([0, arrow_side / 8, arrow_side - arrow_side / 4, 0])
    contact_center_arrow_tip_pose_left_side = np.array([0, - arrow_side / 8, arrow_side - arrow_side / 4, 0])

    rot_mat = np.matmul(np.matmul(rot_z, rot_y), rot_x)

    arrow_tip_pose = np.matmul(rot_mat, contact_center_arrow_tip_pose)
    arrow_tip_right_side_pose = np.matmul(rot_mat, contact_center_arrow_tip_pose_right_side)
    arrow_tip_left_side_pose = np.matmul(rot_mat, contact_center_arrow_tip_pose_left_side)

    end_arrow_point = np.array(contact_pos) + arrow_tip_pose[:3]
    end_arrow_right_side_point = np.array(contact_pos) + arrow_tip_right_side_pose[:3]
    end_arrow_left_side_point = np.array(contact_pos) + arrow_tip_left_side_pose[:3]

    return end_arrow_point, end_arrow_right_side_point, end_arrow_left_side_point


def draw_heat_arrow(contact_pos, fit, end_eff_or_euler, min_fit, max_fit, env, init_obj_xyzw):

    # Sample color based on fit
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    rgba = cmap(fit)
    contact_obj_color = list(rgba)
    i_alpha = 3
    contact_obj_color[i_alpha] = fit

    # Get arrow coordinates
    end_arrow_point, end_arrow_right_side_point, end_arrow_left_side_point = get_arrow_tip_pose(
        contact_pos=contact_pos,
        end_eff_or_euler=end_eff_or_euler,
    )

    line_color = contact_obj_color[:3]
    line_width = cvt_fit2linewidth(fit=fit, max_fit=max_fit, min_fit=min_fit)

    env.p.addUserDebugLine(
        lineFromXYZ=contact_pos,
        lineToXYZ=end_arrow_point,
        lineColorRGB=line_color,
        lineWidth=line_width
    )
    env.p.addUserDebugLine(
        lineFromXYZ=end_arrow_right_side_point,
        lineToXYZ=end_arrow_point,
        lineColorRGB=line_color,
        lineWidth=line_width
    )
    env.p.addUserDebugLine(
        lineFromXYZ=end_arrow_left_side_point,
        lineToXYZ=end_arrow_point,
        lineColorRGB=line_color,
        lineWidth=line_width
    )

    draw_success_archive_cell = False
    if draw_success_archive_cell:
        contact_obj_id = generate_obj_id(env=env, info_shape_contact=info_shape_contact, color=contact_obj_color)
        env.p.resetBasePositionAndOrientation(contact_obj_id, contact_pos, init_obj_xyzw)


def display_success_archive_3d_arrows_heatmap(env, all_eff_pos_touch, fitnesses, all_end_eff_or_euler):

    env = init_env(cfg, display=True)

    env.reset(load='state')
    stabilize_simulation(env)
    env.p.removeBody(env.robot_id)

    all_contact_points = all_eff_pos_touch

    init_obj_pos, init_obj_xyzw = env.p.getBasePositionAndOrientation(env.obj_id)
    min_fit, max_fit = np.quantile(fitnesses, q=0.2), np.quantile(fitnesses, q=0.8)

    for contact_pos, fit, end_eff_or_euler in zip(all_contact_points, fitnesses, all_end_eff_or_euler):

        draw_heat_arrow(
            contact_pos=contact_pos,
            fit=fit,
            end_eff_or_euler=end_eff_or_euler,
            min_fit=min_fit,
            max_fit=max_fit,
            env=env,
            init_obj_xyzw=init_obj_xyzw
        )

    input(f"{len(all_contact_points)} individuals successfully plotted. Press enter to close...")
    env.close()


def skip_non_reproducible_inds(i_inds2skip, success_archive_cell_poses, fitnesses, all_end_eff_or_euler):
    assert len(success_archive_cell_poses) == len(fitnesses) == len(all_end_eff_or_euler)
    success_archive_cell_poses = [
        pose for i_ind, pose in enumerate(success_archive_cell_poses) if i_ind not in i_inds2skip
    ]
    fitnesses = [
        fit for i_ind, fit in enumerate(fitnesses) if i_ind not in i_inds2skip
    ]
    all_end_eff_or_euler = [
        or_euler for i_ind, or_euler in enumerate(all_end_eff_or_euler) if i_ind not in i_inds2skip
    ]
    assert len(success_archive_cell_poses) == len(fitnesses) == len(all_end_eff_or_euler)
    return success_archive_cell_poses, fitnesses, all_end_eff_or_euler


def replay_trajs():
    print(f'Displaying running output.')

    if len(INDIVIDUALS) == 0:
        print(f'folder={folder} no individuals to display. (Is -os triggered ? Is there success ?). Ending.')
        return

    print(f'{len(INDIVIDUALS)} individuals have been found.')

    success_archive_cell_poses = get_map_cells(inds_bds=BEHAVIOR_DESCRIPTORS)
    assert len(FITNESSES) == len(success_archive_cell_poses)

    all_end_eff_or_euler, i_inds2skip = extract_trajectories(individuals=INDIVIDUALS)

    success_archive_cell_poses, fitnesses, all_end_eff_or_euler = skip_non_reproducible_inds(
        i_inds2skip=i_inds2skip,
        success_archive_cell_poses=success_archive_cell_poses,
        fitnesses=FITNESSES,
        all_end_eff_or_euler=all_end_eff_or_euler,
    )

    display_success_archive_3d_arrows_heatmap(
        env=ENV, all_eff_pos_touch=success_archive_cell_poses, fitnesses=fitnesses,
        all_end_eff_or_euler=all_end_eff_or_euler
    )


def main():
    replay_trajs()


if __name__ == "__main__":
    sys.exit(main())


