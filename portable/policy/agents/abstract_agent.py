import contextlib
from contextlib import contextmanager
from abc import ABCMeta, abstractmethod
from typing import Any


class Agent(object, metaclass=ABCMeta):
    """Abstract agent class."""

    training = True

    @abstractmethod
    def act(self, obs: Any) -> Any:
        """Select an action.
        Returns:
            ~object: action
        """
        raise NotImplementedError()

    @abstractmethod
    def observe(self, obs: Any, reward: float, done: bool, reset: bool) -> None:
        """Observe consequences of the last action.
        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, dirname: str) -> None:
        """Save internal states.
        Returns:
            None
        """
        pass

    @abstractmethod
    def load(self, dirname: str) -> None:
        """Load internal states.
        Returns:
            None
        """
        pass

    @contextlib.contextmanager
    def eval_mode(self):
        orig_mode = self.training
        try:
            self.training = False
            yield
        finally:
            self.training = orig_mode


@contextmanager
def evaluating(net: Agent):
    """Temporarily switch to evaluation mode."""
    istrain = net.training
    try:
        net.training = False
        yield net
    finally:
        net.training = istrain
