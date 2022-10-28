"""
collect screen to train initiation and termination datasets
"""
import numpy as np
import matplotlib.pyplot as plt

from . import AntBoxEnv, AntBridgeEnv, AntGoalEnv


def _place_ant_and_save_img(env, x, y, save_dir, imshow=False):
    pos = (x, y)
    print(pos)
    x, y = env.place_ant(pos)
    img = env.render_camera(imshow=False)
    if imshow:
        plt.imsave(save_dir + '/{}_{}.png'.format(x, y), img)
    else:
        np.save(save_dir + f"/{x}_{y}.npy", img)
    return x, y, img


def collect_from_ant_box(save_dir):
    """
    for the original environment, ant starts from (0, 0)
    the whole space spans from x = [-9, 9], y = [-9, 25]
    the box spans from x = [-1.5, 1.5], y = [4, 14], should avoid this area in general

    for initiation classifier:
        the only positive examples are the ones that are right in front of the box,
        x = [-1.2, 0.8], y = [-2, 2]
        NOTE: need to hand import the positive examples of ant on the box, obtained
        by running a successful policy.
        all others are negative examples
    for termination classifier:
        all pictures on the other side of the box are positive examples, while those
        on the starting side are negative.
    """
    env = AntBoxEnv()
    o = env.reset()

    # initiation positive examples
    sub_dir = os.path.join(save_dir, 'initiation', 'positive')
    os.makedirs(sub_dir)
    for x in np.linspace(-1.2, 0.8, 4):
        for y in np.linspace(-2, 2, 6):
            _place_ant_and_save_img(env, x, y, sub_dir)

    # initiation negative examples
    sub_dir = os.path.join(save_dir, 'initiation', 'negative')
    os.makedirs(sub_dir)
    # left half of x
    for x in np.linspace(-9, -2, 7):
        for y in np.linspace(-9, 25, 34):
            _place_ant_and_save_img(env, x, y, sub_dir)
    # middle x -- avoid the box in y
    for x in np.linspace(-2, 2, 4):
        for y in np.linspace(-9, -4, 5):
            _place_ant_and_save_img(env, x, y, sub_dir)
        for y in np.linspace(14, 25, 11):
            _place_ant_and_save_img(env, x, y, sub_dir)
    # right half of x
    for x in np.linspace(2, 9, 7):
        for y in np.linspace(-9, 25, 34):
            _place_ant_and_save_img(env, x, y, sub_dir)
    
    # termination positive examples
    sub_dir = os.path.join(save_dir, 'termination', 'positive')
    os.makedirs(sub_dir)
    for x in np.linspace(-9, 9, 18):
        for y in np.linspace(15, 25, 10):
            _place_ant_and_save_img(env, x, y, sub_dir)
    
    # termination negative examples
    sub_dir = os.path.join(save_dir, 'termination', 'negative')
    os.makedirs(sub_dir)
    for x in np.linspace(-9, 9, 18):
        for y in np.linspace(-9, 10, 19):
            if -2 < x < 2 and -4 < y < 8:
                continue
            _place_ant_and_save_img(env, x, y, sub_dir)


def collect_from_ant_bridge(save_dir):
    """
    The original env spans x = [-9, 9], y = [-2, 27]
    The bridge spans x = [-1, 2] y = [4, 22]
    NOTE: the x position of the bridge is dependent on seed=4

    for initiation classifier:
        positive examples are those right in front of the bridge and on the bridge
        all others are negative examples
    for termination classifier:
        positive examples are all those on the other side, and all on
        the bridge and on this side are negative examples

    there is a large gap beteen the bridge and the ground, so 
    we teleport the ant to two different y-blocks
    """
    env = AntBridgeEnv()
    o = env.reset()

    # initiation positive examples
    sub_dir = os.path.join(save_dir, 'initiation', 'positive')
    os.makedirs(sub_dir)
    for x in np.linspace(0.3, 1.2, 4):
        for y in np.linspace(-2, 20, 22):
            _place_ant_and_save_img(env, x, y, sub_dir)
    
    # initiation negative examples
    sub_dir = os.path.join(save_dir, 'initiation', 'negative')
    os.makedirs(sub_dir)
    # left half of x -- avoid the gap in y
    for x in np.linspace(-9, -1, 8):
        for y in np.linspace(-2, 4, 6):
            _place_ant_and_save_img(env, x, y, sub_dir)
        for y in np.linspace(22, 27, 5):
            _place_ant_and_save_img(env, x, y, sub_dir)
    # middle x -- avoid the bridge in y
    for x in np.linspace(-1, 2, 4):
        for y in np.linspace(21, 27, 6):
            _place_ant_and_save_img(env, x, y, sub_dir)
    # right half of x -- avoid the gap in y
    for x in np.linspace(2, 9, 7):
        for y in np.linspace(-2, 4, 6):
            _place_ant_and_save_img(env, x, y, sub_dir)
        for y in np.linspace(22, 27, 5):
            _place_ant_and_save_img(env, x, y, sub_dir)

    # termination positive examples
    sub_dir = os.path.join(save_dir, 'termination', 'positive')
    os.makedirs(sub_dir)
    for x in np.linspace(-9, 9, 18):
        for y in np.linspace(22, 27, 5):
            _place_ant_and_save_img(env, x, y, sub_dir)
    
    # termination negative examples
    sub_dir = os.path.join(save_dir, 'termination', 'negative')
    os.makedirs(sub_dir)
    # on the bridge
    for x in np.linspace(0.3, 1.2, 4):
        for y in np.linspace(4, 20, 16):
            _place_ant_and_save_img(env, x, y, sub_dir)
    # on this side
    for x in np.linspace(-9, 9, 18):
        for y in np.linspace(-2, 4, 6):
            _place_ant_and_save_img(env, x, y, sub_dir)


def collect_fron_ant_goal(save_dir):
    """
    the original env spans x = [-9, 9], y = [0, 28]
    The wall spans y = [8, 12], should avoid this range
    NOTE: with seed=4, the goal is at x = [-1, 1]

    for initiation classifier:
        positive examples are those crossing the doorway, all others are negative examples
    for termination classifier:
        positive examples are those on the other side, all others are negative examples
    """
    env = AntGoalEnv()
    o = env.reset()

    # initiation positive examples
    sub_dir = os.path.join(save_dir, 'initiation', 'positive')
    os.makedirs(sub_dir)
    for x in np.linspace(-0.5, 0.5, 4):
        for y in np.linspace(7, 11, 4):
            _place_ant_and_save_img(env, x, y, sub_dir)
    
    # initiation negative examples
    sub_dir = os.path.join(save_dir, 'initiation', 'negative')
    os.makedirs(sub_dir)
    # on the other side
    for x in np.linspace(-9, 9, 18):
        for y in np.linspace(12, 28, 16):
            _place_ant_and_save_img(env, x, y, sub_dir)
    # on this side -- avoid the area in front of the goal
    for x in np.linspace(-9, -1, 8):
        for y in np.linspace(0, 8, 8):
            _place_ant_and_save_img(env, x, y, sub_dir)
    for x in np.linspace(-1, 1, 2):
        for y in np.linspace(0, 6, 6):
            _place_ant_and_save_img(env, x, y, sub_dir)
    for x in np.linspace(1, 9, 8):
        for y in np.linspace(0, 8, 8):
            _place_ant_and_save_img(env, x, y, sub_dir)
    
    # termination positive examples
    sub_dir = os.path.join(save_dir, 'termination', 'positive')
    os.makedirs(sub_dir)
    for x in np.linspace(-9, 9, 18):
        for y in np.linspace(12, 28, 16):
            _place_ant_and_save_img(env, x, y, sub_dir)
    
    # termination negative examples
    sub_dir = os.path.join(save_dir, 'termination', 'negative')
    os.makedirs(sub_dir)
    # in doorway 
    for x in np.linspace(-0.5, 0.5, 4):
        for y in np.linspace(8, 10, 2):
            _place_ant_and_save_img(env, x, y, sub_dir)
    # on this side 
    for x in np.linspace(-9, 9, 18):
        for y in np.linspace(0, 8, 8):
            _place_ant_and_save_img(env, x, y, sub_dir)


if __name__ == "__main__":
    import os
    import argparse
    from skills.utils import create_log_dir

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='box')
    parser.add_argument('--save_dir', type=str, default='results/classifier_data')
    parser.add_argument('--seed', type=int, default=4)
    args = parser.parse_args()

    # deterministic
    np.random.seed(args.seed)

    # create saving dir based on env
    save_dir = os.path.join(args.save_dir, args.env)
    create_log_dir(save_dir, remove_existing=True, log_git=False)

    if args.env == 'box':
        collect_from_ant_box(save_dir)
    elif args.env == 'bridge':
        collect_from_ant_bridge(save_dir)
    elif args.env == 'goal':
        collect_fron_ant_goal(save_dir)

