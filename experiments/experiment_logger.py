import matplotlib.pyplot as plt 
import os
from glob import glob
import numpy as np
import moviepy.video.io.ImageSequenceClip

class VideoGenerator():
    def __init__(self,
                 save_path) -> None:
        self.save_path = save_path 
        self.temp_path = os.path.join(save_path,"tmp")
        
        self.text = ""
        self.clear_images()
        os.makedirs(self.save_path, exist_ok=True)
        
    def episode_start(self):
        self.clear_images()
        os.makedirs(self.temp_path, exist_ok=True)
    
    def episode_end(self, video_name):
        try:
            self.make_video(video_name)
        except:
            print("Video creation failed. Skipping for this episode....")
        self.clear_images()
    
    def add_line(self, line):
        self.text += line
        self.text += "\n"
    
    def clear_images(self):
        if os.path.exists(self.temp_path):
            for file in glob(os.path.join(self.temp_path, "*.png")):
                os.remove(file)
            
            os.rmdir(self.temp_path)
    
    def make_image(self,
                   img):
        img = np.squeeze(img)
        # change channels to align with rgb layout
        # img = np.transpose(img, (1,2,0))
        
        i = 0
        while os.path.exists(os.path.join(self.temp_path, "{}.png".format(i))):
            i += 1

        fig, (ax1, ax2) = plt.subplots(1,2)
        
        ax1.imshow(img)
        ax2.text(0,1, self.text, horizontalalignment='left', verticalalignment='top', fontsize=6, transform=ax2.transAxes)
        
        ax1.axis('off')
        ax2.axis('off')
        
        fig.savefig(os.path.join(self.temp_path, "{}.png".format(i)))
        
        plt.close(fig)
        self.text = ""
        
        
    def make_video(self, video_name):
        images = []
        
        for file in sorted(glob(os.path.join(self.temp_path, '*.png')), 
                           key=lambda f: int(''.join(filter(str.isdigit, f)))):
            images.append(file)
        
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(images, fps=1)
        
        video_file = os.path.join(self.save_path, "{}.mp4".format(video_name))
        
        clip.write_videofile(video_file)
        