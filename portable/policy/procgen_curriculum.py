"""
based on looking at the game layout in procgen interactive mode, 
I arranged the level-seeds in an order that is roughly increasing in difficulty
All games are only observed for seeds 0-19
"""

caveflyer_levels = [
    0, 9, 11, 14, 15,  # easy: no enemies or rocks, just flying
    5, 8, 2, 12, 10, 16, 17,18, 19, 3, 7, 13,  # medium: enemies and rocks
    1, 4, 6  # hard: enemies that are difficult to avoid or long traversal distance
]


coinrun_levels = [
    3, 12, 18, 2, 17, 13, 0,  # easy: no enemies, or simple enemies
    19, 16, 9, 15, 5, 14, 1,  # medium: enemies but the goal-distance is not too long
    11, 8, 6, 10, 4, 7,       # hard: hard enemies and long traversal distance
]


procgen_game_curriculum = {
    'caveflyer': caveflyer_levels,
    'coinrun': coinrun_levels,
}