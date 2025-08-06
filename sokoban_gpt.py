import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy
import os, openai
from dotenv import load_dotenv
# from padded_levels import levels
from normal_levels import levels as normal_levels

def load_level():
    global level, WIDTH, HEIGHT
    level = copy.deepcopy(normal_levels[current_level])
    cell_size = 23
    level_width = max(len(row) for row in level)
    level_height = len(level)
    WIDTH = level_width * cell_size
    HEIGHT = level_height * cell_size

def on_key_down(key):
    global current_level

    if key in [keys.UP, keys.DOWN, keys.LEFT, keys.RIGHT]:
        for test_y, row in enumerate(level):
            for test_x, cell in enumerate(row):
                if cell == player or cell == player_on_storage:
                    player_x = test_x
                    player_y = test_y

        dx = 0
        dy = 0
        if key == keys.LEFT:
            dx = -1
        elif key == keys.RIGHT:
            dx = 1
        elif key == keys.UP:
            dy = -1
        elif key == keys.DOWN:
            dy = 1

        current = level[player_y][player_x]
        adjacent = level[player_y + dy][player_x + dx]
        beyond = ''
        if (
            0 <= player_y + dy + dy < len(level)
            and 0 <= player_x + dx + dx < len(level[player_y + dy + dy])
        ):
            beyond = level[player_y + dy + dy][player_x + dx + dx]

        next_adjacent = {
            empty: player,
            storage: player_on_storage,
        }
        next_current = {
            player: empty,
            player_on_storage: storage,
        }
        next_beyond = {
            empty: box,
            storage: box_on_storage,
        }
        next_adjacent_push = {
            box: player,
            box_on_storage: player_on_storage,
        }

        if adjacent in next_adjacent:
            level[player_y][player_x] = next_current[current]
            level[player_y + dy][player_x + dx] = next_adjacent[adjacent]

        elif beyond in next_beyond and adjacent in next_adjacent_push:
            level[player_y][player_x] = next_current[current]
            level[player_y + dy][player_x + dx] = next_adjacent_push[adjacent]
            level[player_y + dy + dy][player_x + dx + dx] = next_beyond[beyond]

        complete = True

        for y, row in enumerate(level):
            for x, cell in enumerate(row):
                if cell == box:
                    complete = False

        if complete:
            current_level += 1

            if current_level >= len(normal_levels):
                current_level = 0

            load_level()

    elif key == keys.R:
        load_level()

    elif key == keys.N:
        current_level += 1
        if current_level >= len(normal_levels):
            current_level = 0
        load_level()

    elif key == keys.P:
        current_level -= 1
        if current_level < 0:
            current_level = len(normal_levels) - 1
        load_level()

    # Press 'G' to request a single GPT move manually
    elif key == keys.G:
        key_const = gpt_next_move()
        if key_const:
            # Re-use the normal movement logic by calling this handler recursively
            # with the arrow key returned by GPT.
            on_key_down(key_const)

def gpt_next_move():
    """Ask GPT-4o-mini for the next move and return a keys.* constant or None."""
    board_text = board_as_text()
    # DEBUG: show the exact board that will be sent to GPT
    print("Sending board to GPT:\n" + board_text)

    global conversation_history

    # Summarise recent assistant moves (for loop-avoidance guidance)
    recent_assistant = [m["content"] for m in conversation_history if m["role"] == "assistant"]
    last_moves = ", ".join(recent_assistant[-5:]) if recent_assistant else "none"

    # Identify player and unsolved box coordinates
    p_x, p_y = player_coord()
    boxes = unsolved_boxes()
    boxes_str = ", ".join(f"({x},{y})" for x, y in boxes) if boxes else "none"

    # Build and add new user message
    user_msg = {
        "role": "user",
        "content": (
            f"Previous moves: {last_moves}\n"
            f"Player: ({p_x},{p_y})\n"
            f"Unsolved boxes: {boxes_str}\n"
            f"Board:\n{board_text}\nYour move?"
        ),
    }
    messages = conversation_history + [user_msg]

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=240,
        )
        # answer = response.choices[0].message.content.strip().lower()
        # print(f"GPT raw answer: {answer!r}")  # DEBUG in terminal
        full_answer = response.choices[0].message.content.strip()
        print("GPT says:\n", full_answer)       # lets you read its plan

        # Update conversation history with the exchange
        assistant_msg = {"role": "assistant", "content": full_answer}
        conversation_history += [user_msg, assistant_msg]

        # Trim history to keep the last 30 turns (60 msgs) plus the system prompt
        MAX_TURNS = 30
        excess = len(conversation_history) - (1 + MAX_TURNS * 2)
        if excess > 0:
            # Preserve the first (system) message
            conversation_history = [conversation_history[0]] + conversation_history[1 + excess:]

        # Extract a move token (up/down/left/right) from the assistant's reply
        answer_token = None
        for word in reversed(full_answer.lower().split()):
            if word in MOVE_TOKENS:
                answer_token = word
                break
        if answer_token:
            return MOVE_TOKENS[answer_token]

        print("Unrecognised GPT answer – no move applied.")
        return None
    except Exception as exc:
        print("OpenAI error:", exc)
        return None

def draw():
    screen.fill((255, 255, 190))

    for y, row in enumerate(level):
        for x, cell in enumerate(row):
            if cell != empty:
                cell_size = 23

                colors = {
                    player: (167, 135, 255),
                    player_on_storage: (158, 119, 255),
                    box: (255, 201, 126),
                    box_on_storage: (150, 255, 127),
                    storage: (156, 229, 255),
                    wall: (255, 147, 209),
                }

                screen.draw.filled_rect(
                    Rect(
                        (x * cell_size, y * cell_size),
                        (cell_size, cell_size)
                    ),
                    colors[cell]
                )

                screen.draw.text(
                    cell,
                    (x * cell_size, y * cell_size),
                    color=(255, 255, 255)
                )

def player_coord():
    """Return (x, y) coordinate of the player (@ or +)."""
    for y, row in enumerate(level):
        for x, cell in enumerate(row):
            if cell in (player, player_on_storage):
                return (x, y)
    return (-1, -1)  # should not happen


def unsolved_boxes():
    """Return list of (x, y) coordinates of boxes ($) not yet on storage."""
    coords = []
    for y, row in enumerate(level):
        for x, cell in enumerate(row):
            if cell == box:  # '$' means box not on storage
                coords.append((x, y))
    return coords


def board_as_text():
    """Return the current level grid as an ASCII multiline string."""
    return "\n".join("".join(row) for row in level)

def drive_gpt():
    """Periodic callback that fetches a move from GPT and applies it."""
    if not AUTO_PLAY:
        return
    key_const = gpt_next_move()
    if key_const:
        on_key_down(key_const)

load_dotenv()

# Initialise key variables before loading the first level
current_level = 0
openai.api_key = os.getenv("OPENAI_API_KEY")

# Now that current_level is defined we can safely load the starting level
load_level()

player = '@'
player_on_storage = '+'
box = '$'
box_on_storage = '*'
storage = '.'
wall = '#'
empty = ' '

# Mapping from GPT textual responses to Pygame Zero key constants
MOVE_TOKENS = {
    "up": keys.UP,
    "down": keys.DOWN,
    "left": keys.LEFT,
    "right": keys.RIGHT,
}

SYSTEM_PROMPT = (
    "You are a Sokoban-solver AI.\n"
    "ASCII legend: @ player, + on storage, $ box, * stored box, . storage, # wall.\n"
    "Coordinates: (x,y) using 0-based indexing with (0,0) at the top-left.\n"
    "If you do up: y-1, down: y+1, left: x-1, right: x+1.\n"
    "Goal: push every $ onto a . so all boxes become *.\n"
    "Think step-by-step: briefly explain the move, then finish with just one of up, down, left, right on its own line.\n"
    # "Reply with ONE word: up, down, left, or right.\n"
    # "Prefer moves that push a box into empty storage.\n"
)

# Conversation history (system message plus successive user/assistant turns)
conversation_history = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
]

# Toggle to enable/disable GPT autoplay (manual by default to save tokens)
AUTO_PLAY = False

# Start the periodic query loop only if autoplay is enabled
if AUTO_PLAY:
    clock.schedule_interval(drive_gpt, 1.0)
