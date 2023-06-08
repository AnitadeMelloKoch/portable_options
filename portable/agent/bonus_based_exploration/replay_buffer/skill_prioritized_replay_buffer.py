from dopamine.replay_memory import prioritized_replay_buffer
from dopamine.replay_memory import circular_replay_buffer
from dopamine.replay_memory import sum_tree
from dopamine.replay_memory.circular_replay_buffer import ReplayElement
from portable.agent.bonus_based_exploration.replay_buffer import skill_circular_replay_buffer

import numpy as np
import collections
import gin.tf
import tensorflow as tf

@gin.configurable
class OutOfGraphPrioritizedSkillReplayBuffer(skill_circular_replay_buffer.OutOfGraphSkillReplayBuffer):

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
        super().__init__(observation_shape, 
                         stack_size, 
                         replay_capacity, 
                         batch_size, 
                         mask_shape, 
                         mask_dtype, 
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

        self.sum_tree = sum_tree.SumTree(replay_capacity)

    def get_add_args_signature(self):
        """The signature of the add function.
        The signature is the same as the one for OutOfGraphReplayBuffer, with an
        added priority.
        Returns:
        list of ReplayElements defining the type of the argument signature needed
            by the add function.
        """
        parent_add_signature = super(
            OutOfGraphPrioritizedSkillReplayBuffer,
            self).get_add_args_signature()
        add_signature = parent_add_signature + [
            ReplayElement('priority', (), np.float32)
        ]

        return add_signature

    def _add(self, *args):
        """Internal add method to add to the underlying memory arrays.
        The arguments need to match add_arg_signature.
        If priority is none, it is set to the maximum priority ever seen.
        Args:
        *args: All the elements in a transition.
        """
        self._check_args_length(*args)

        # Use Schaul et al.'s (2015) scheme of setting the priority of new elements
        # to the maximum priority so far.
        # Picks out 'priority' from arguments and adds it to the sum_tree.
        transition = {}
        for i, element in enumerate(self.get_add_args_signature()):
            if element.name == 'priority':
                priority = args[i]
            else:
                transition[element.name] = args[i]

        self.sum_tree.set(self.cursor(), priority)
        super(OutOfGraphPrioritizedSkillReplayBuffer, self)._add_transition(transition)

    def sample_index_batch(self, batch_size):
        """Returns a batch of valid indices sampled as in Schaul et al. (2015).
        Args:
        batch_size: int, number of indices returned.
        Returns:
        list of ints, a batch of valid indices sampled uniformly.
        Raises:
        Exception: If the batch was not constructed after maximum number of tries.
        """
        # Sample stratified indices. Some of them might be invalid.
        indices = self.sum_tree.stratified_sample(batch_size)
        allowed_attempts = self._max_sample_attempts
        for i in range(len(indices)):
            if not self.is_valid_transition(indices[i]):
                if allowed_attempts == 0:
                    raise RuntimeError(
                        'Max sample attempts: Tried {} times but only sampled {}'
                        ' valid indices. Batch size is {}'.
                        format(self._max_sample_attempts, i, batch_size))
                index = indices[i]
                while not self.is_valid_transition(index) and allowed_attempts > 0:
                    # If index i is not valid keep sampling others. Note that this
                    # is not stratified.
                    index = self.sum_tree.sample()
                    allowed_attempts -= 1
                indices[i] = index
        return indices

    def sample_transition_batch(self, batch_size=None, indices=None):
        """Returns a batch of transitions with extra storage and the priorities.
        The extra storage are defined through the extra_storage_types constructor
        argument.
        When the transition is terminal next_state_batch has undefined contents.
        Args:
        batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
        indices: None or list of ints, the indices of every transition in the
            batch. If None, sample the indices uniformly.
        Returns:
        transition_batch: tuple of np.arrays with the shape and type as in
            get_transition_elements().
        """
        transition = (super(OutOfGraphPrioritizedSkillReplayBuffer, self).
                        sample_transition_batch(batch_size, indices))
        transition_elements = self.get_transition_elements(batch_size)
        transition_names = [e.name for e in transition_elements]
        probabilities_index = transition_names.index('sampling_probabilities')
        indices_index = transition_names.index('indices')
        indices = transition[indices_index]
        # The parent returned an empty array for the probabilities. Fill it with the
        # contents of the sum tree.
        transition[probabilities_index][:] = self.get_priority(indices)
        
        return transition

    def set_priority(self, indices, priorities):
        """Sets the priority of the given elements according to Schaul et al.
        Args:
        indices: np.array with dtype int32, of indices in range
            [0, replay_capacity).
        priorities: float, the corresponding priorities.
        """
        assert indices.dtype == np.int32, ('Indices must be integers, '
                                        'given: {}'.format(indices.dtype))
        # Convert JAX arrays to NumPy arrays first, since it is faster to iterate
        # over the entirety of a NumPy array than a JAX array.
        priorities = np.asarray(priorities)
        for index, priority in zip(indices, priorities):
            self.sum_tree.set(index, priority)

    def get_priority(self, indices):
        """Fetches the priorities correspond to a batch of memory indices.
        For any memory location not yet used, the corresponding priority is 0.
        Args:
        indices: np.array with dtype int32, of indices in range
            [0, replay_capacity).
        Returns:
        priorities: float, the corresponding priorities.
        """
        assert indices.shape, 'Indices must be an array.'
        assert indices.dtype == np.int32, ('Indices must be int32s, '
                                        'given: {}'.format(indices.dtype))
        batch_size = len(indices)
        priority_batch = np.empty((batch_size), dtype=np.float32)
        for i, memory_index in enumerate(indices):
            priority_batch[i] = self.sum_tree.get(memory_index)
        return priority_batch

    def get_transition_elements(self, batch_size=None):
        """Returns a 'type signature' for sample_transition_batch.
        Args:
        batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
        Returns:
        signature: A namedtuple describing the method's return type signature.
        """
        parent_transition_type = (
            super(OutOfGraphPrioritizedSkillReplayBuffer,
                self).get_transition_elements(batch_size))
        probablilities_type = [
            ReplayElement('sampling_probabilities', (batch_size,), np.float32)
        ]
        return parent_transition_type + probablilities_type

@gin.configurable(
    denylist=['observation_shape', 'stack_size', 'update_horizon', 'gamma'])
class WrappedPrioritizedSkillReplayBuffer(skill_circular_replay_buffer.WrappedSkillReplayBuffer):

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

        if wrapped_memory is None:
            wrapped_memory = OutOfGraphPrioritizedSkillReplayBuffer(
                observation_shape, stack_size, replay_capacity, batch_size,
                mask_shape, mask_dtype, update_horizon, gamma, max_sample_attempts,
                extra_storage_types=extra_storage_types,
                observation_dtype=observation_dtype)

        super().__init__(observation_shape, 
                         stack_size, 
                         mask_shape, 
                         mask_dtype, 
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

    def tf_set_priority(self, indices, priorities):
        """Sets the priorities for the given indices.
        Args:
        indices: tf.Tensor with dtype int32 and shape [n].
        priorities: tf.Tensor with dtype float and shape [n].
        Returns:
        A tf op setting the priorities for prioritized sampling.
        """
        return tf.numpy_function(
            self.memory.set_priority, [indices, priorities], [],
            name='prioritized_replay_set_priority_py_func')

    def tf_get_priority(self, indices):
        """Gets the priorities for the given indices.
        Args:
        indices: tf.Tensor with dtype int32 and shape [n].
        Returns:
        priorities: tf.Tensor with dtype float and shape [n], the priorities at
            the indices.
        """
        return tf.numpy_function(
            self.memory.get_priority, [indices],
            tf.float32,
            name='prioritized_replay_get_priority_py_func')


