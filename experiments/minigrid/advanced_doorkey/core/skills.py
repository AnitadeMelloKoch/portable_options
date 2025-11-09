from typing import List, Union, Tuple
import gymnasium as gym
from gymnasium.core import Env, Wrapper
from minigrid.core.actions import Actions as MG
import numpy as np
from copy import deepcopy

# directions might be flipped (might not be moving in the direction)
# check for whether you're in the right room, otherwise use Astar. 
# want to have an initiation + termination set. convert these into a class which is a 
# skill class which each skill has a policy.
# leave move_towards in this env. and each class can have a termination set which is a function. Give each an initation
# and termination set. give it a function that says whether or not we are already in the termination set.
# moving things into a skill, initation set, policy, and termination set.
# write a wrapper over whole environment that is the new skill environment. each skill is accessible within that class
# and uses the internal environment

# if finish doing it, want to train with these skills. PPO agent trained it to use minigrid but now train it on skills.
# runs it on -1. use GPU to -1.

COLOR_TO_RGB = {
    "red":    (255,   0,   0),
    "green":  (0,   255,   0),
    "blue":   (0,     0, 255),
    "yellow": (255, 255,   0),
    "purple": (160,  32, 240),
    "grey":   (128, 128, 128),
}

def make_monochrome_state(
    state: Union[np.ndarray, dict],
    color: Union[str, Tuple[int, int, int]]
) -> np.ndarray:
    if isinstance(state, dict):
        if "image" in state:
            img = state["image"]
        elif "obs" in state:
            img = state["obs"]
        else:
            raise ValueError("State dict must have 'image' or 'obs' key.")
    else:
        img = state

    img = np.asarray(img, dtype=np.float32)

    chw = False
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
        chw = True

    if img.max() <= 1.0:
        img = img * 255.0

    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    if isinstance(color, str):
        if color not in COLOR_TO_RGB:
            raise ValueError(f"Unknown color name: {color}")
        rgb = COLOR_TO_RGB[color]
    else:
        rgb = color

    color_vec = np.asarray(rgb, dtype=np.float32) / 255.0
    tinted = gray[..., None] * color_vec
    tinted = np.clip(tinted, 0, 255).astype(np.uint8)

    if chw:
        tinted = np.transpose(tinted, (2, 0, 1))

    return tinted

class actions:
    LEFT = MG.left
    RIGHT = MG.right
    FORWARD = MG.forward
    PICKUP = MG.pickup
    DROP = MG.drop
    TOGGLE = MG.toggle

class Skill:
    name: str
    def can_start(self, env) -> bool: ...
    def is_done(self, env) -> bool: ...
    def tick(self, env) -> None: ...

class GetKey(Skill):
    name = "get_key"
    def can_start(self, env): return env.objects["key"] is not None
    def is_done(self, env): return env.objects["key"] is None
    def tick(self, env):
        k = env.objects["key"]
        if k is None: raise StopIteration
        ax, ay = env.get_current_position()
        kx, ky = k.position
        if abs(ax-kx)+abs(ay-ky) == 1:
            env._face_cell((kx,ky))
            env._env_step(actions.PICKUP)
            if self.is_done(env): raise StopIteration
            return
        env.move_towards((kx,ky), want_adjacent=True)

class OpenDoor(Skill):
    name = "open_door"
    def can_start(self, env): 
        return env.objects["door"] is not None and not env.get_door_obj().is_open
    def is_done(self, env): 
        return env.objects["door"] is not None and env.get_door_obj().is_open
    def tick(self, env):
        d = env.objects["door"]
        if d is None: raise StopIteration
        dx, dy = d.position
        ax, ay = env.get_current_position()
        if abs(ax-dx)+abs(ay-dy) == 1:
            env._face_cell((dx,dy))
            env._env_step(actions.TOGGLE)
            if self.is_done(env): raise StopIteration
            return
        env.move_towards((dx,dy), want_adjacent=True)

class CloseDoor(Skill):
    name = "close_door"
    def can_start(self, env): return env.objects["door"] is not None and env.get_door_obj().is_open
    def is_done(self, env): return env.objects["door"] is not None and not env.get_door_obj().is_open
    def tick(self, env):
        d = env.objects["door"]
        if d is None: raise StopIteration
        dx, dy = d.position
        ax, ay = env.get_current_position()
        if abs(ax-dx)+abs(ay-dy) == 1:
            env._face_cell((dx,dy))
            env._env_step(actions.TOGGLE)
            if self.is_done(env): raise StopIteration
            return
        env.move_towards((dx,dy), want_adjacent=True)

class GoToGoal(Skill):
    name = "go_to_goal"
    def can_start(self, env):
        if env.objects["goal"] is None: return False
        return env.get_door_obj().is_open or env._same_room_as_goal()
    def is_done(self, env):
        if env.objects["goal"] is None: return False
        return tuple(env.get_current_position()) == tuple(env.objects["goal"])
    def tick(self, env):
        gx, gy = env.objects["goal"]
        if self.is_done(env): raise StopIteration
        env.move_towards((gx,gy), want_adjacent=False)

class DropKey(Skill):
    name = "drop_key"
    def can_start(self, env):
        c = getattr(env.env.unwrapped, "carrying", None)
        return env.objects["key"] is None and c is not None
    def is_done(self, env): return env.objects["key"] is not None
    def tick(self, env):
        c = getattr(env.env.unwrapped, "carrying", None)
        if c is None: raise StopIteration
        env._env_step(actions.DROP)
        raise StopIteration

class SkillEnv(Wrapper):
    def __init__(self, env: Env, option_reward=1.0, max_skill_horizon=200, door_colour=None):
        super().__init__(env)
        self.option_reward = option_reward
        self.max_skill_horizon = max_skill_horizon
        self.door_colour = door_colour
        self.objects = {"key": None, "door": None, "goal": None}
        self.obs = None
        self._timestep = 0
        self.skills: List[Skill] = [GetKey(), OpenDoor(), CloseDoor(), GoToGoal(), DropKey()]
        self.current_color = None
        self._env_done = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs = deepcopy(obs)
        self._find_objs()
        self._set_door_colour()
        self._timestep = 0
        self._env_done = False
        return obs, self._info()

    def step(self, skill_idx: int):
        sk = self.skills[int(skill_idx)]
        color_names = list(COLOR_TO_RGB.keys())
        self.current_color = color_names[int(skill_idx) % len(color_names)]
        if not sk.can_start(self): return deepcopy(self.obs), 0.0, False, {"blocked": True, "skill": sk.name}
        total_r = 0.0
        self._env_done = False
        for _ in range(self.max_skill_horizon):
            if sk.is_done(self) or self._env_done:
                break
            try:
                sk.tick(self)
            except StopIteration:
                pass
            if sk.is_done(self) or self._env_done:
                break
        done = sk.is_done(self)
        r = self.option_reward if done else 0.0
        total_r += r
        return deepcopy(self.obs), total_r, self._env_done, {"skill": sk.name, "terminated": done}

    def get_current_position(self):
        p = self.env.unwrapped.agent_pos
        return (int(p[0]), int(p[1]))


    def _env_step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, r, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            obs, r, done, info = result
        if action == actions.FORWARD and self.env.unwrapped.agent_dir == 3: 
            tint_color = "red"
        elif action == actions.FORWARD and self.env.unwrapped.agent_dir == 1:  
            tint_color = "blue"
        elif self.env.unwrapped.agent_dir == 0:   
            return "green"
        elif self.env.unwrapped.agent_dir == 2:  
            return "yellow"
        else:
            tint_color = None

        if tint_color is not None:
            obs = make_monochrome_state(obs, tint_color)
        self.obs = deepcopy(obs)
        self._timestep += 1
        self._find_objs()
        if done:
            self._env_done = True
        return obs, r, done, info

    def _is_free(self, x, y):
        if not (0 <= x < self.env.unwrapped.width and 0 <= y < self.env.unwrapped.height): return False
        cell = self.env.unwrapped.grid.get(x, y)
        if cell is None: return True
        if hasattr(cell, "can_overlap") and callable(cell.can_overlap): return cell.can_overlap()
        if cell.type == "door" and getattr(cell, "is_open", False): return True
        return False

    def _neighbors(self, x, y):
        for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if self._is_free(nx, ny): yield (nx, ny)

    def _same_room_as_goal(self):
        if self.objects["goal"] is None: return False
        ax, ay = self.get_current_position()
        gx, gy = self.objects["goal"]
        return self._astar((ax, ay), (gx, gy)) is not None

    def _face_cell(self, target):
        tx, ty = target
        ax, ay = self.get_current_position()
        if (tx, ty) == (ax+1, ay): desired = 0
        elif (tx, ty) == (ax-1, ay): desired = 2
        elif (tx, ty) == (ax, ay+1): desired = 1
        else: desired = 3
        while self.env.unwrapped.agent_dir != desired:
            if (self.env.unwrapped.agent_dir - desired) % 4 == 1:
                self._env_step(actions.LEFT)
            else:
                self._env_step(actions.RIGHT)

    def step_toward_cell(self, target):
        self._face_cell(target)
        self._env_step(actions.FORWARD)

    def move_towards(self, target_pos, want_adjacent=False):
        ax, ay = self.get_current_position()
        tx, ty = target_pos
        if want_adjacent:
            cand = [(tx+1,ty),(tx-1,ty),(tx,ty+1),(tx,ty-1)]
            cand = [(x,y) for (x,y) in cand if 0 <= x < self.env.unwrapped.width and 0 <= y < self.env.unwrapped.height and self._is_free(x,y)]
            if not cand:
                self._env_step(np.random.choice([actions.LEFT, actions.RIGHT])); self._env_step(actions.FORWARD); return
            paths = []
            for cx, cy in cand:
                p = self._astar((ax,ay),(cx,cy))
                if p is not None and len(p) >= 2: paths.append(p)
            if not paths:
                self._env_step(np.random.choice([actions.LEFT, actions.RIGHT])); self._env_step(actions.FORWARD); return
            path = min(paths, key=len)
        else:
            path = self._astar((ax,ay),(tx,ty))
            if path is None or len(path) < 2:
                self._env_step(np.random.choice([actions.LEFT, actions.RIGHT])); self._env_step(actions.FORWARD); return
        self.step_toward_cell(path[1])

    def _astar(self, start, goal):
        import heapq
        sx, sy = start; gx, gy = goal
        h = lambda x,y: abs(x-gx)+abs(y-gy)
        openq = [(h(sx,sy), 0, (sx,sy), None)]
        came = {}; gscore = {(sx,sy): 0}; closed = set()
        while openq:
            _, g, cur, parent = heapq.heappop(openq)
            if cur in closed: continue
            came[cur] = parent
            if cur == (gx,gy):
                path = []; c = cur
                while c is not None: path.append(c); c = came[c]
                path.reverse(); return path
            closed.add(cur)
            cx, cy = cur
            for nx, ny in self._neighbors(cx, cy):
                ng = g + 1
                if (nx,ny) in closed: continue
                if ng < gscore.get((nx,ny), 1e9):
                    gscore[(nx,ny)] = ng
                    heapq.heappush(openq, (ng + h(nx,ny), ng, (nx,ny), cur))
        return None

    def _find_objs(self):
        self.objects = {"key": None, "door": None, "goal": None}
        for x in range(self.env.unwrapped.width):
            for y in range(self.env.unwrapped.height):
                cell = self.env.unwrapped.grid.get(x, y)
                if not cell: continue
                if cell.type == "key" and self.objects["key"] is None:
                    self.objects["key"] = type("KeyTuple", (), {"position": (x,y), "colour": getattr(cell, "color", None)})
                elif cell.type == "door" and self.objects["door"] is None:
                    self.objects["door"] = type("DoorTuple", (), {"position": (x,y), "colour": getattr(cell, "color", None)})
                elif cell.type == "goal" and self.objects["goal"] is None:
                    self.objects["goal"] = (x, y)

    def get_door_obj(self):
        d = self.objects["door"]
        if d is None: return None
        x, y = d.position
        return self.env.unwrapped.grid.get(x, y)

    def _set_door_colour(self):
        if self.door_colour is None: return

    def _info(self):
        return {"timestep": self._timestep, "key": self.objects["key"], "door": self.objects["door"], "goal": self.objects["goal"], "agent_pos": self.get_current_position()}
    
    
# example usage
if __name__ == "__main__":
    import gymnasium as gym
    base = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array")
    env = SkillEnv(base, option_reward=1.0)
    obs, info = env.reset()
   
    for a in [0,1,3]:
        obs, r, done, info = env.step(a)
