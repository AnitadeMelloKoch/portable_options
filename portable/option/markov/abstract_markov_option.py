from abc import ABCMeta, abstractmethod
import logging
from typing import Any
import logging

logger = logging.getLogger(__name__)

"""
Base markov option class.

A Markov Option represents a single possible instance of an option
execution. This is to be used for verification of an instance
which can be used for association and assimilation.
"""

class MarkovOption(object, metaclass=ABCMeta):

    def __init__(self,
            use_log=True):
        self.use_log = use_log

    def log(self, message):
        if self.use_log:
            logger.info(message)

    @abstractmethod
    def save(self, path:str):
        pass

    @abstractmethod
    def load(self, path:str):
        pass

    @abstractmethod
    def can_initiate(
            self, 
            agent_space_state: Any):
        """
        Check Markov Option initiation condition
        Input:
            agent_space_state: state to be evaluated.
        Return:
            bool: True if can initiate False if can't initiate
        """
        raise NotImplementedError()

    @abstractmethod
    def can_terminate(
            self, 
            agent_space_state: Any):
        """
        Check Markov Option termination condition
        Input:
            agent_space_state: state to be evaluated.
        Return:
            bool: True if can terminate False if can't terminate
        """
        raise NotImplementedError()

    @abstractmethod
    def interact_initiation(
            self, 
            positive_agent_space_states: list, 
            negative_agent_space_states: list):
        """
        Provide feedback to initiation classifier.
        Input:
            positive_agent_space_states: list of positive samples, if any,
                for this instance of the Option.
            negative_agent_space_states: list of negative samples, if any,
                for this instance of the Option.
        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def interact_termination(
            self,
            positive_agent_space_states: list,
            negative_agent_space_states: list):
        """
        Provide feedback to termination classifier.
        Input:
            positive_agent_space_states: list of positive samples, if any,
                for this instance of the Option.
            negative_agent_space_states: list of negative samples, if any,
                for this instance of the Option.
        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def run(self,
            env: Any,
            state: Any,
            info: dict,
            evaluate: bool):
        """
        Run this option instance to completion or failure.
        Input:
            env: environment instance
            state: agent's current state obs
            info: info from current state
            evaluate: run in evaluation mode
        Return:
            state: agent's current state obs at end of run()
            total_reward: total reward accumulated during run()
            done: bool True if episode is over, False if episode is not over
            info: agent's current state info
            steps: number of steps taken in run()

        """
        raise NotImplementedError()

