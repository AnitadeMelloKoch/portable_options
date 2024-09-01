import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

class MontePlotter():
    def __init__(self,
                 plot_dir) -> None:
        self.locations_init = defaultdict(list)
        self.locations_term = defaultdict(list)
        self.plot_dir = plot_dir
    
    def record_init_location(self,
                             action,
                             location):
        self.locations_init[action].append(location)
    
    def record_term_location(self,
                             action,
                             location):
        self.locations_term[action].append(location)
    
    def _plot(self, 
              locations,
              plot_dir_name):
        room_x = defaultdict(list)
        room_y = defaultdict(list)
        for action in locations.keys():
            for location in locations[action]:
                room_x[location[2]].append(location[0])
                room_y[location[2]].append(location[1])

            for room in room_x.keys():
                plot_dir = os.path.join(self.plot_dir,
                                        plot_dir_name)
                os.makedirs(plot_dir, exist_ok=True)
                plot_name = os.path.join(plot_dir,
                                         "action{}_room{}.png".format(action, room))
                
                fig = plt.figure(num=1)
                ax = fig.add_subplot()
                ax.set_ylim([0, 300])
                ax.set_xlim([0, 160])
                ax.scatter(room_x[room], room_y[room])
                fig.savefig(plot_name)
                plt.close(fig)
    
    def plot(self, plot_dir_name):
        self._plot(self.locations_init,
                   os.path.join(plot_dir_name, "init"))
        self._plot(self.locations_term,
                   os.path.join(plot_dir_name, "term"))
    
    def reset(self):
        self.locations_init = defaultdict(list)
        self.locations_term = defaultdict(list)
    
