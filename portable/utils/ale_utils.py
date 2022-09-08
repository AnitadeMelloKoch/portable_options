from collections import defaultdict


def _getIndex(address):
    """
    helper function for parsing ram address
    get the index of the ram address using teh row and column format
    """
    assert type(address) == str and len(address) == 2
    row, col = tuple(address)
    row = int(row, 16) - 8
    col = int(col, 16)
    return row * 16 + col


def getByte(ram, address):
    """Return the byte at the specified emulator RAM location"""
    idx = _getIndex(address)
    return ram[idx]


def get_player_position(ram):
    """
    given the ram state, get the position of the player
    """
    # return the player position at a particular state
    x = int(getByte(ram, 'aa'))
    y = int(getByte(ram, 'ab'))
    return x, y


def get_skull_position(ram):
    """
    given the ram state, get the x position of the skull
    """
    x = int(getByte(ram, 'af'))
    level = 0
    screen = get_player_room_number(ram)
    skull_offset = defaultdict(lambda: 33, {
        18: [1,23,12][level],
    })[screen]
    # Note: up to some rounding, player dies when |player_x - skull_x| <= 6
    return x + skull_offset


def get_level(ram):
    return int(getByte(ram, 'b9'))


def get_object_position(ram):
    x = int(getByte(ram, 'ac'))
    y = int(getByte(ram, 'ad'))
    return x, y


def get_in_air(ram):
    # jump: 255 is on the ground, when initiating jump, turns to 16, 12, 8, 4, 0, 255, ...
    jump = getByte(ram, 'd6')
    # fall: 0 is on the groud, even positive numbers are falling (jumping is not included)
    fall = getByte(ram, 'd8')
    return jump != 255, fall > 0


def set_player_position(env, x, y):
    """
    set the player position, specifically made for monte envs
    """
    state_ref = env.unwrapped.ale.cloneState()
    state = env.unwrapped.ale.encodeState(state_ref)
    env.unwrapped.ale.deleteState(state_ref)

    state[331] = x
    state[335] = y

    new_state_ref = env.unwrapped.ale.decodeState(state)
    env.unwrapped.ale.restoreState(new_state_ref)
    env.unwrapped.ale.deleteState(new_state_ref)
    env.step(0)  # NO-OP action to update the RAM state


def get_player_room_number(ram):
    """
    given the ram state, get the room number of the player
    """
    return int(getByte(ram, '83'))


def set_player_ram(env, ram_state):
    """
    completely override the ram with a saved ram state
    """
    state_ref = env.unwrapped.ale.cloneState()
    env.unwrapped.ale.deleteState(state_ref)
    
    new_state_ref = env.unwrapped.ale.decodeState(ram_state)
    env.unwrapped.ale.restoreState(new_state_ref)
    env.unwrapped.ale.deleteState(new_state_ref)
    obs, _, _, _ = env.step(0)  # NO-OP action to update the RAM state
    return obs
