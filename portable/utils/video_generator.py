import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import glob
import pickle

class VideoGenerator():

    def __init__(self, base_path):
        self.base_path = base_path
        self.img_path = os.path.join(self.base_path, 'tmp')
        # self.clear_images()

    def save_frame(self, img, title):
        # img = np.squeeze(img)
        os.makedirs(self.img_path, exist_ok=True)

        i = 0
        while os.path.exists(os.path.join(self.img_path, "{}.pkl".format(i))):
            i += 1

        save_path = os.path.join(self.img_path, "{}.pkl".format(i))

        frame = {
            "image": img,
            "title": title
        }

        with open(save_path, "wb") as f:
            pickle.dump(frame, f)

    def save_image(self, fig):
        os.makedirs(self.img_path, exist_ok=True)

        i = 0
        while os.path.exists(os.path.join(self.img_path, "{}.png".format(i))):
            i += 1

        save_path = os.path.join(self.img_path, "{}.png".format(i))

        fig.savefig(save_path)

    def create_video_from_images(self, video_file_name):
        
        video_file_path = os.path.join(self.base_path, video_file_name)

        frames = []

        for file in sorted(
            glob.glob(os.path.join(self.img_path, '*.png')),
            key=lambda f:int(''.join(filter(str.isdigit,f)))
        ):
            frames.append(plt.imread(file))

        plt.clf()
        fig = plt.figure(num=1, clear=True)
        ax = fig.add_subplot()
        ax.axis('off')
        patch = ax.imshow(frames[0])

        def animate(idx):
            frame = frames[idx]
            patch.set_data(frame)
            fig.tight_layout()

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=100)
        anim.save(video_file_path, writer=animation.FFMpegWriter(fps=5))

        plt.close(fig)
        plt.clf()
        self.clear_images()

    def create_video(self, video_file_name, save_path=None):
        if save_path is None:
            save_path = self.base_path
        video_file_path = os.path.join(save_path, video_file_name)

        frames = []

        for file in sorted(
            glob.glob(os.path.join(self.img_path, '*.pkl')),
            key=lambda f:int(''.join(filter(str.isdigit,f)))
        ):
            with open(file, "rb") as f:
                frames.append(pickle.load(f))

        plt.clf()
        fig = plt.figure(num=1, clear=True)
        ax = fig.add_subplot()
        ax.axis('off')
        patch = ax.imshow(frames[0]["image"])
        ax.set_title(frames[0]["title"])

        def animate(idx):
            frame = frames[idx]
            patch.set_data(frame["image"])
            ax.set_title(frame["title"])
            fig.tight_layout()

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=100)
        anim.save(video_file_path, writer=animation.FFMpegWriter(fps=5))

        plt.close(fig)
        plt.clf()
        self.clear_images()

    def clear_images(self):
        if os.path.exists(self.img_path):
            for file in glob.glob(os.path.join(self.img_path, "*")):
                os.remove(file)

            os.rmdir(self.img_path)
