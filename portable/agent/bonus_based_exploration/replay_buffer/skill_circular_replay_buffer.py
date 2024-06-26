from dopamine.replay_memory import circular_replay_buffer

import numpy as np
import gin.tf
import collections

ReplayElement = circular_replay_buffer.ReplayElement

@gin.configurable
class OutOfGraphSkillReplayBuffer(circular_replay_buffer.OutOfGraphReplayBuffer):

    """
    The base out of graph replay buffer stores single observations and stacks these
    at runtime to remove inefficiencies from storing several copies of single frames.
    For skill based executions, we need to store the stack from the final option step.
    """

    def __init__(self, 
            observation_shape, 
            stack_size,
            replay_capacity, 
            batch_size, 
            mask_shape,
            mask_dtype,
            update_horizon=1, 
            gamma=0.99, 
            max_sample_attempts=1000, 
            extra_storage_types=None, 
            observation_dtype=np.uint8, 
            terminal_dtype=np.uint8, 
            action_shape=(), 
            action_dtype=np.int32, 
            reward_shape=(), 
            reward_dtype=np.float32, 
            checkpoint_duration=4, 
            keep_every=None):
        
        self.mask_shape = mask_shape
        self.mask_dtype = mask_dtype
        self._state_stack_size = stack_size
        
        super().__init__(
                observation_shape, 
                1, # stack_size 
                replay_capacity, 
                batch_size, 
                update_horizon, 
                gamma, 
                max_sample_attempts, 
                extra_storage_types, 
                observation_dtype, 
                terminal_dtype, 
                action_shape, 
                action_dtype, 
                reward_shape, 
                reward_dtype, 
                checkpoint_duration, 
                keep_every)

        
        self._state_shape = self._observation_shape + (self._state_stack_size,)

    def get_transition_elements(self, batch_size=None):
        """Returns a 'type signature' for sample_transition_batch.
        Args:
        batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
        Returns:
        signature: A namedtuple describing the method's return type signature.
        """
        batch_size = self._batch_size if batch_size is None else batch_size

        transition_elements = [
            ReplayElement('state', (batch_size,) + self._observation_shape
                         + (self._state_stack_size,), self._observation_dtype),
            ReplayElement('action', (batch_size,) + self._action_shape,
                        self._action_dtype),
            ReplayElement('reward', (batch_size,) + self._reward_shape,
                        self._reward_dtype),
            ReplayElement('next_state', (batch_size,) + self._observation_shape
                         + (self._state_stack_size,), self._observation_dtype),
            ReplayElement('next_action', (batch_size,) + self._action_shape,
                        self._action_dtype),
            ReplayElement('next_reward', (batch_size,) + self._reward_shape,
                        self._reward_dtype),
            ReplayElement('mask', (batch_size,) + self.mask_shape,
                        self.mask_dtype),
            ReplayElement('terminal', (batch_size,), self._terminal_dtype),
            ReplayElement('indices', (batch_size,), np.int32)
        ]
        for element in self._extra_storage_types:
            transition_elements.append(
                ReplayElement(element.name, (batch_size,) + tuple(element.shape),
                                element.type))

        return transition_elements

    def sample_transition_batch(self, batch_size=None, indices=None):
        """Returns a batch of transitions (including any extra contents).

        If get_transition_elements has been overridden and defines elements not
        stored in self._store, an empty array will be returned and it will be
        left to the child class to fill it. For example, for the child class
        OutOfGraphPrioritizedReplayBuffer, the contents of the
        sampling_probabilities are stored separately in a sum tree.

        When the transition is terminal next_state_batch has undefined contents.

        NOTE: This transition contains the indices of the sampled elements. These
        are only valid during the call to sample_transition_batch, i.e. they may
        be used by subclasses of this replay buffer but may point to different data
        as soon as sampling is done.

        Args:
        batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
        indices: None or list of ints, the indices of every transition in the
            batch. If None, sample the indices uniformly.

        Returns:
        transition_batch: tuple of np.arrays with the shape and type as in
            get_transition_elements().

        Raises:
        ValueError: If an element to be sampled is missing from the replay buffer.
        """

        if batch_size is None:
            batch_size = self._batch_size
        if indices is None:
            indices = self.sample_index_batch(batch_size)
        assert len(indices) == batch_size

        transition_elements = self.get_transition_elements(batch_size)
        batch_arrays = self._create_batch_arrays(batch_size)
        for batch_element, state_index in enumerate(indices):
            trajectory_indices = [(state_index + j) % self._replay_capacity
                                    for j in range(self._update_horizon)]
            trajectory_terminals = self._store['terminal'][trajectory_indices]
            is_terminal_transition = trajectory_terminals.any()
            if not is_terminal_transition:
                trajectory_length = self._update_horizon
            else:
                # np.argmax of a bool array returns the index of the first True.
                trajectory_length = np.argmax(trajectory_terminals.astype(bool),
                                            0) + 1
            next_state_index = state_index + trajectory_length
            trajectory_discount_vector = (
                self._cumulative_discount_vector[:trajectory_length])
            trajectory_rewards = self.get_range(self._store['reward'], state_index,
                                                next_state_index)

            # Fill the contents of each array in the sampled batch.
            assert len(transition_elements) == len(batch_arrays)
            for element_array, element in zip(batch_arrays, transition_elements):
                if element.name == 'state':
                    element_array[batch_element] = self._store[element.name][state_index]
                elif element.name == 'reward':
                # compute the discounted sum of rewards in the trajectory.
                    element_array[batch_element] = np.sum(
                        trajectory_discount_vector * trajectory_rewards, axis=0)
                elif element.name in ('next_action', 'next_reward', 'next_state'):
                    element_array[batch_element] = (
                        self._store[element.name.lstrip('next_')][(next_state_index) %
                                                                self._replay_capacity])
                elif element.name == 'terminal':
                    element_array[batch_element] = is_terminal_transition
                elif element.name == 'indices':
                    element_array[batch_element] = state_index
                elif element.name == 'mask':
                    element_array[batch_element] = (
                        self._store[element.name][(next_state_index) %
                                                    self._replay_capacity])
                elif element.name in self._store.keys():
                    element_array[batch_element] = (
                        self._store[element.name][state_index])
                # We assume the other elements are filled in by the subclass.

        return batch_arrays

    def get_storage_signature(self):
        """Returns a default list of elements to be stored in this replay memory.

        Note - Derived classes may return a different signature.

        Returns:
        list of ReplayElements defining the type of the contents stored.
        """
        storage_elements = [
            ReplayElement('state', self._observation_shape
                    + (self._state_stack_size,), self._observation_dtype),
            ReplayElement('action', self._action_shape, self._action_dtype),
            ReplayElement('reward', self._reward_shape, self._reward_dtype),
            ReplayElement('terminal', (), self._terminal_dtype),
            ReplayElement('mask', self.mask_shape, self.mask_dtype)
        ]

        for extra_replay_element in self._extra_storage_types:
            storage_elements.append(extra_replay_element)
        return storage_elements

    def add(self, 
            state,
            action, 
            reward, 
            terminal, 
            *args, 
            priority=None, 
            episode_end=False):

        if priority is not None:
            args = args + (priority,)

        self._check_add_types(state, action, reward, terminal, *args)
        if self._next_experience_is_episode_start:
            for _ in range(self._stack_size - 1):
                self._add_zero_transition()
        self._next_experience_is_episode_start = False
        
        if episode_end or terminal:
            self.episode_end_indices.add(self.cursor())
            self._next_experience_is_episode_start = True
        else:
            self.episode_end_indices.discard(self.cursor())

        self._add(state, action, reward, terminal, *args)

@gin.configurable(
    denylist=['observation_shape', 'stack_size','update_horizon', 'gamma']
)
class WrappedSkillReplayBuffer(circular_replay_buffer.WrappedReplayBuffer):
    """Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism.

    Usage:
        To add a transition:  call the add function.

        To sample a batch:    Construct operations that depend on any of the
                            tensors is the transition dictionary. Every sess.run
                            that requires any of these tensors will sample a new
                            transition.
    """
    
    def __init__(self, 
                 observation_shape, 
                 stack_size, 
                 mask_shape,
                 mask_dtype,
                 use_staging=False, 
                 replay_capacity=1000000, 
                 batch_size=32, 
                 update_horizon=1, 
                 gamma=0.99, 
                 wrapped_memory=None, 
                 max_sample_attempts=1000, 
                 extra_storage_types=None, 
                 observation_dtype=np.uint8, 
                 terminal_dtype=np.uint8, 
                 action_shape=(), 
                 action_dtype=np.int32, 
                 reward_shape=(), 
                 reward_dtype=np.float32):
        """Initializes WrappedReplayBuffer.

        Args:
        observation_shape: tuple of ints.
        stack_size: int, number of frames to use in state stack.
        use_staging: bool, when True it would use a staging area to prefetch
            the next sampling batch.
        replay_capacity: int, number of transitions to keep in memory.
        batch_size: int.
        update_horizon: int, length of update ('n' in n-step update).
        gamma: int, the discount factor.
        wrapped_memory: The 'inner' memory data structure. If None,
            it creates the standard DQN replay memory.
        max_sample_attempts: int, the maximum number of attempts allowed to
            get a sample.
        extra_storage_types: list of ReplayElements defining the type of the extra
            contents that will be stored and returned by sample_transition_batch.
        observation_dtype: np.dtype, type of the observations. Defaults to
            np.uint8 for Atari 2600.
        terminal_dtype: np.dtype, type of the terminals. Defaults to np.uint8 for
            Atari 2600.
        action_shape: tuple of ints, the shape for the action vector. Empty tuple
            means the action is a scalar.
        action_dtype: np.dtype, type of elements in the action.
        reward_shape: tuple of ints, the shape of the reward vector. Empty tuple
            means the reward is a scalar.
        reward_dtype: np.dtype, type of elements in the reward.

        Raises:
        ValueError: If update_horizon is not positive.
        ValueError: If discount factor is not in [0, 1].
        """
        if wrapped_memory is None:
            wrapped_memory = OutOfGraphSkillReplayBuffer(
                observation_shape,
                stack_size,
                replay_capacity,
                batch_size,
                mask_shape,
                mask_dtype,
                update_horizon,
                gamma,
                max_sample_attempts,
                observation_dtype=observation_dtype,
                terminal_dtype=terminal_dtype,
                extra_storage_types=extra_storage_types,
                action_shape=action_shape,
                action_dtype=action_dtype,
                reward_shape=reward_shape,
                reward_dtype=reward_dtype
            )
        super().__init__(observation_shape, 
                         stack_size, 
                         use_staging, 
                         replay_capacity, 
                         batch_size, 
                         update_horizon, 
                         gamma, 
                         wrapped_memory, 
                         max_sample_attempts, 
                         extra_storage_types, 
                         observation_dtype, 
                         terminal_dtype, 
                         action_shape, 
                         action_dtype, 
                         reward_shape, 
                         reward_dtype)
    
    def add(self, state, action, reward, terminal, *args):
        """Adds a transition to the replay memory.

        Since the next_observation in the transition will be the observation added
        next there is no need to pass it.

        If the replay memory is at capacity the oldest transition will be discarded.

        Args:
        observation: list of np.array each with shape observation_shape.
        action: int, the action in the transition.
        reward: float, the reward received in the transition.
        terminal: np.dtype, acts as a boolean indicating whether the transition
                    was terminal (1) or not (0).
        *args: extra contents with shapes and dtypes according to
            extra_storage_types.
        """
        return super().add(state, action, reward, terminal, *args)


    def unpack_transition(self, transition_tensors, transition_type):
        """Unpacks the given transition into member variables.
        Args:
        transition_tensors: tuple of tf.Tensors.
        transition_type: tuple of ReplayElements matching transition_tensors.
        """
        self.transition = collections.OrderedDict()
        for element, element_type in zip(transition_tensors, transition_type):
            self.transition[element_type.name] = element

        # TODO(bellemare): These are legacy and should probably be removed in
        # future versions.
        self.states = self.transition['state']
        self.actions = self.transition['action']
        self.rewards = self.transition['reward']
        self.next_states = self.transition['next_state']
        self.next_actions = self.transition['next_action']
        self.next_rewards = self.transition['next_reward']
        self.terminals = self.transition['terminal']
        self.indices = self.transition['indices']
        self.mask = self.transition['mask']

