import numpy as np
from experiments.divdis_minigrid.core.advanced_minigrid_mock_terminations import BaseTermination

def check_box_in_place(num_required, info):
    if 'in_target_box_locations' in info:
        return len(info["in_target_box_locations"]) == num_required
    return False

class OneBoxPlaced(BaseTermination):
    def check_term(self, state, env):
        info = env.get_info()
        return check_box_in_place(1, info)

class TwoBoxPlaced(BaseTermination):
    def check_term(self, state, env):
        info = env.get_info()
        return check_box_in_place(2, info)

class ThreeBoxPlaced(BaseTermination):
    def check_term(self, state, env):
        info = env.get_info()
        return check_box_in_place(3, info)
