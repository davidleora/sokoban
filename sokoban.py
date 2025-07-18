import copy

from padded_levels import levels

current_level = 0


def load_level():
    global level, WIDTH, HEIGHT
    level = copy.deepcopy(levels[current_level])
    cell_size = 23
    level_width = max(len(row) for row in level)
    level_height = len(level)
    WIDTH = level_width * cell_size
    HEIGHT = level_height * cell_size


load_level()

player = '@'
player_on_storage = '+'
box = '$'
box_on_storage = '*'
storage = '.'
wall = '#'
empty = ' '


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

            if current_level >= len(levels):
                current_level = 0

            load_level()

    elif key == keys.R:
        load_level()

    elif key == keys.N:
        current_level += 1
        if current_level >= len(levels):
            current_level = 0
        load_level()

    elif key == keys.P:
        current_level -= 1
        if current_level < 0:
            current_level = len(levels) - 1
        load_level()


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
