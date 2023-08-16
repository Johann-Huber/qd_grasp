

ROBOT_KWARGS = {
    'kuka_ik': {
        'gym_env_name': 'kuka_grasping-v0',
        'gene_per_keypoints': 6,
        'link_id_contact': None,
        'nb_steps_to_rollout': 10,
        'nb_iter_ref': 2000,
        'n_it_closing_grip': 25,
        'controller_type': 'interpolate keypoints cartesian speed control grip',
    },

    'kuka_iiwa_allegro_ik': {
        'gym_env_name': 'kuka_iiwa_allegro-v0',
        'gene_per_keypoints': 6,  # cartesian : 6-DOF [x, y, z, theta_1, theta_2, theta_3]
        'link_id_contact': None,
        'nb_steps_to_rollout': 10,
        'nb_iter_ref': 1500,
        'n_it_closing_grip': 50,
        'controller_type': 'interpolate keypoints cartesian finger synergies',
    },
}


