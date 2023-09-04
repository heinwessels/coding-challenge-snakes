from random import choice
from typing import List, Tuple

import math
import numpy as np
from enum import Enum, auto

from ...bot import Bot
from ...constants import Move, MOVE_VALUE_TO_DIRECTION
from ...snake import Snake

sign = lambda x: math.copysign(1, x)

def is_on_grid(pos: np.array, grid_size: Tuple[int, int]) -> bool:
    """
    Check if a position is still on the grid
    """
    return 0 <= pos[0] < grid_size[0] and 0 <= pos[1] < grid_size[1]


def collides(pos: np.array, snakes: List[Snake]) -> bool:
    """
    Check if a position is occupied by any of the snakes
    """
    for snake in snakes:
        if snake.collides(pos):
            return True
    return False

def vectors_to_candies(head : np.array, candies: List[np.array]) -> List[np.array]:
    directions = []
    for candy in candies:
        angle = math.atan2(candy[1]-head[1], candy[0] - head[0])
        directions.append(np.array([math.cos(angle), math.sin(angle)]))
    return directions

def vector_to_move(vector : np.array) -> Move:
    if abs(vector[0]) >  abs(vector[1]):
        return Move.RIGHT if vector[0] > 0 else Move.LEFT
    else:
        return Move.UP if vector[1] > 0 else Move.DOWN
    
def angle_to_move(angle : float, zig_zag : bool) -> Move:
    # Wrap angle
    if angle < 0:
        angle += 2 * math.pi

    # Diagonal angles need to zig zag
    if angle / (math.pi / 2) > 0.01 and zig_zag:
        angle += math.pi / 2

    # Determine which move works best
    return {
            0: Move.RIGHT,
            1: Move.UP,
            2: Move.LEFT,
            3: Move.DOWN
        }[(angle // (math.pi / 2)) % 4]

class Mood(Enum):
    HUNGRY = auto()
    ANNOYING = auto()
    CENTERING = auto()

class Annoyance:
    figured_out = False
    primary_direction = None
    secondary_direction = None
    
    bored = 50
    duration = 0

class ApologeticApophis(Bot):

    def __init__(self, id: int, grid_size: Tuple[int, int]):
        super(ApologeticApophis, self).__init__(id, grid_size)
        self.mood = Mood.HUNGRY
        self.annoyance = Annoyance()
        self.zig_zag = False
        self.center = (np.array(grid_size) + 1) / 2

    @property
    def name(self):
        return 'Apologetic Apophis'

    @property
    def contributor(self):
        return 'Hein'

    def determine_next_move(self, snake: Snake, other_snakes: List[Snake], candies: List[np.array]) -> Move:
        self.snake = snake
        self.head = snake[0]
        self.other_snake = other_snakes[0]
        self.candies = candies
        self.zig_zag = not self.zig_zag

        if self.mood == Mood.HUNGRY and self._should_start_annoying(snake, other_snakes):
            print("Starting centering")
            self.mood = Mood.CENTERING
    
        moves = self._determine_possible_moves(snake, other_snakes[0])
        return self.choose_move(moves, snake, other_snakes, candies)

    def _should_start_annoying(self, snake: Snake, other_snakes: List[Snake]):
        minimim_length = 9
        if len(snake) > minimim_length:
            return True
        return False

    def _determine_possible_moves(self, snake, other_snake) -> List[Move]:
        """
        Return a list with all moves that we want to do. Later we'll choose one from this list randomly. This method
        will be used during unit-testing
        """
        # highest priority, a move that is on the grid
        on_grid = [move for move in MOVE_VALUE_TO_DIRECTION
                   if is_on_grid(snake[0] + MOVE_VALUE_TO_DIRECTION[move], self.grid_size)]
        if not on_grid:
            return list(Move)

        # then avoid collisions with other snakes
        collision_free = [move for move in on_grid
                          if not collides(snake[0] + MOVE_VALUE_TO_DIRECTION[move], [snake, other_snake])]
        if collision_free:
            return collision_free
        else:
            return on_grid
        
    def _figure_out_annoyance(self):
        print("Figuring out")

        other_snake_head = self.other_snake[0]
        angle_to_other_snake = math.atan2(other_snake_head[1]-self.head[1], other_snake_head[0] - self.head[0])
        snapped_angle = round(angle_to_other_snake / (math.pi/4)) * (math.pi/4)
        angle_delta = angle_to_other_snake - snapped_angle
        
        self.annoyance.primary_direction = snapped_angle + sign(angle_delta) * math.pi / 2
        self.annoyance.secondary_direction = angle_to_other_snake

        self.annoyance.figured_out = True
        self.annoyance.duration = 0

    def choose_move(self, moves: List[Move], snake : Snake, other_snakes : List[Snake], candies) -> Move:

        if self.mood == Mood.CENTERING:
            to_center_vector = self.center - self.head
            if sum(abs(to_center_vector)) < 2:
                print("Gonna be annoying")
                self.mood = Mood.ANNOYING
            else:
                move = vector_to_move(to_center_vector)
                if move in moves:
                    return move

        if self.mood == Mood.ANNOYING:
            if not self.annoyance.figured_out:
                    self._figure_out_annoyance()
            else:
                self.annoyance.duration += 1
                if self.annoyance.duration == self.annoyance.bored:
                    self.mood = Mood.CENTERING

            annoyance_move = angle_to_move(self.annoyance.primary_direction, self.zig_zag)
            
            if not is_on_grid(self.head + MOVE_VALUE_TO_DIRECTION[annoyance_move], self.grid_size):
                self.annoyance.primary_direction -= math.pi
                if self.annoyance.primary_direction < 0:
                    self.annoyance.primary_direction += 2 * math.pi
                annoyance_move = angle_to_move(self.annoyance.primary_direction, self.zig_zag)

            if annoyance_move in moves:
                return annoyance_move
            print("No move")

        vectors = vectors_to_candies(snake[0], candies)
        weights = {move:-100 for move in moves}
        for move in weights.keys():
            for vector in vectors:
                if move == vector_to_move(vector):
                    if max(vector) > weights[move]:
                        weights[move] = max(vector)
        return max(weights, key=weights.get)