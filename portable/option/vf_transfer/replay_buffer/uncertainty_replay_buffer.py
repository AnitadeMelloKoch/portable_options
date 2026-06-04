import random
from dataclasses import dataclass, field
from typing import Any
from sortedcontainers import SortedList


@dataclass(order=True)
class Transition:
    uncertainty: float
    state: Any = field(compare=False)
    next_state: Any = field(compare=False)
    reward: float = field(compare=False)
    terminal: bool = field(compare=False)
    task_id: Any = field(compare=False)


class UncertaintyReplayBuffer():
    def __init__(self, max_capacity):
        self.max_capacity = max_capacity

        self.memory = SortedList()

    def add_transitions(self, transitions):
        """Add a list of Transition objects."""
        self.memory.update(transitions)
        while len(self.memory) > self.max_capacity:
            self.memory.pop(-1)  # remove highest uncertainty (lowest confidence) first

    def batches(self, batch_size, shuffle=True):
        indices = list(range(len(self.memory)))
        if shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            yield [self.memory[i] for i in indices[start:start + batch_size]]

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        return self.memory[idx]