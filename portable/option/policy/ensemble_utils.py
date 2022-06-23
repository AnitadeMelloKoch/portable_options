import numpy as np
from PIL import Image, ImageFont, ImageDraw 


def pillow_im_add_margin(pil_img, top=0, right=0, bottom=0, left=0, color=0):
    """
    add margin to a pillow image
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def visualize_state_with_ensemble_actions(obs, meaningful_actions, meaningful_q_vals, action_taken, save_path):
    """
    use matplotlib.pyplot and Pillow to visualize the state as well as
    each of the actions chosen by the learners in the ensemble

    args:
        obs: lazy frame object
        meaningful_actions: list of actions as strings
        meaningful_q_vals: list of q-values as strings
        action_taken: action taken by the ensemble, as a string
        save_path: path to the png image
    """
    frame = np.array(obs)[-1]
    image = Image.fromarray(frame.astype(np.uint8))
    image = pillow_im_add_margin(image, left=100, bottom=70)
    
    # write the action and q value on it too
    txt = "\n".join([a + "  Qval: " + q for a, q in zip(meaningful_actions, meaningful_q_vals)])
    txt += "\n\n taken: " + action_taken
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    image_editable = ImageDraw.Draw(image)
    image_editable.text((0, 0), txt, 255, font=font)
    image.save(save_path)


def visualize_state_with_action(obs, action, save_path):
    """
    use matplotlib.pyplot and Pillow to visualize the state as well as the single
    action take by the learner
    This is used for DQN, where only one action is outputed

    args:
        obs: lazy frame object
        action: action taken by the ensemble, as a string
        save_path: path to the png image
    """
    frame = np.array(obs)[-1]
    image = Image.fromarray(frame.astype(np.uint8))
    image = pillow_im_add_margin(image, left=70)
    
    # write the action and q value on it too
    txt = "taken: " + action
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    image_editable = ImageDraw.Draw(image)
    image_editable.text((0, 0), txt, 255, font=font)
    image.save(save_path)
