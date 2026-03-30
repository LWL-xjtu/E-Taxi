from __future__ import annotations

ACTIONS = (
    "stay",
    "accept_order",
    "go_charge",
    "move_north",
    "move_south",
    "move_east",
    "move_west",
)

ACTION_STAY = 0
ACTION_ACCEPT_ORDER = 1
ACTION_GO_CHARGE = 2
ACTION_MOVE_NORTH = 3
ACTION_MOVE_SOUTH = 4
ACTION_MOVE_EAST = 5
ACTION_MOVE_WEST = 6

MODES = (
    "idle",
    "serving",
    "charging",
    "repositioning",
)

MODE_IDLE = 0
MODE_SERVING = 1
MODE_CHARGING = 2
MODE_REPOSITIONING = 3

LOCAL_OBS_DIM = 10
ACTION_DIM = len(ACTIONS)
