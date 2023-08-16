
import utils.constants as consts


def get_simulation_table_height(table_height):
    '''Util function to homogeneize computation of table height : make it both supporting previous setups and the real
    scene.'''

    if consts.REAL_SCENE:
        #  Real scene Y : default (=0.4) - 20 cm
        simulated_table_z = -1 + (table_height - consts.REAL_SCENE_TABLE_HEIGHT)
    else:
        # Default scene (from legacy code)
        simulated_table_z = -1 + (table_height - 0.625)

    return simulated_table_z

