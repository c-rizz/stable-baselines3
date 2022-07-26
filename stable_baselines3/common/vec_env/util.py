"""
Helpers for dealing with vectorized environments.
"""
from collections import OrderedDict
from doctest import UnexpectedException
from typing import Any, Dict, List, Tuple

import gym
import numpy as np

from stable_baselines3.common.preprocessing import check_for_nested_spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs


def copy_obs_dict(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Deep-copy a dict of numpy arrays.

    :param obs: a dict of numpy arrays.
    :return: a dict of copied numpy arrays.
    """
    assert isinstance(obs, OrderedDict), f"unexpected type for observations '{type(obs)}'"
    return OrderedDict([(k, np.copy(v)) for k, v in obs.items()])


def dict_to_obs(obs_space: gym.spaces.Space, obs_dict: Dict[Any, np.ndarray]) -> VecEnvObs:
    """
    Convert an internal representation raw_obs into the appropriate type
    specified by space.

    :param obs_space: an observation space.
    :param obs_dict: a dict of numpy arrays.
    :return: returns an observation of the same type as space.
        If space is Dict, function is identity; if space is Tuple, converts dict to Tuple;
        otherwise, space is unstructured and returns the value raw_obs[None].
    """
    if isinstance(obs_space, gym.spaces.Dict):
        return obs_dict
    elif isinstance(obs_space, gym.spaces.Tuple):
        assert len(obs_dict) == len(obs_space.spaces), "size of observation does not match size of observation space"
        return tuple((obs_dict[i] for i in range(len(obs_space.spaces))))
    else:
        assert set(obs_dict.keys()) == {None}, "multiple observation keys for unstructured observation space"
        return obs_dict[None]


def obs_space_info(obs_space: gym.spaces.Space) -> Tuple[List[str], Dict[Any, Tuple[int, ...]], Dict[Any, np.dtype]]:
    """
    Get dict-structured information about a gym.Space.

    Dict spaces are represented directly by their dict of subspaces.
    Tuple spaces are converted into a dict with keys indexing into the tuple.
    Unstructured spaces are represented by {None: obs_space}.

    :param obs_space: an observation space
    :return: A tuple (keys, shapes, dtypes):
        keys: a list of dict keys.
        shapes: a dict mapping keys to shapes.
        dtypes: a dict mapping keys to dtypes.
    """
    check_for_nested_spaces(obs_space)
    if isinstance(obs_space, gym.spaces.Dict):
        assert isinstance(obs_space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        subspaces = obs_space.spaces
    elif isinstance(obs_space, gym.spaces.Tuple):
        subspaces = {i: space for i, space in enumerate(obs_space.spaces)}
    else:
        assert not hasattr(obs_space, "spaces"), f"Unsupported structured space '{type(obs_space)}'"
        subspaces = {None: obs_space}
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = box.dtype
    return keys, shapes, dtypes


class SpaceInfo():
    def __init__(self, indexes, is_img, shape, dtype):
        self.indexes = indexes
        self.is_img = is_img
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return f"[{self.indexes}, {self.is_img}, {self.shape}, {self.dtype}]"

def _is_img(box_space):
    obs_shape = box_space.shape
    if len(obs_shape)==1:
        return False
    elif len(obs_shape)==2 or len(obs_shape)==3:
        return True
    else:
        raise UnexpectedException(f"Unexpected obs_shape {obs_shape}")

def _img_shape_chw(box_space):
    obs_shape = box_space.shape
    if len(obs_shape)==2:
        # 1-channel image
        return (1, obs_shape[0], obs_shape[1])
    elif len(obs_shape)==3:
        # multi-channel image
        return (obs_shape[0], obs_shape[1], obs_shape[2])
    else:
        raise UnexpectedException(f"Unexpected image obs_shape {obs_shape}")

def _get_space_info(obs_space):
    if isinstance(obs_space, gym.spaces.Box):
        is_img = _is_img(obs_space)
        if is_img:
            shape = _img_shape_chw(obs_space)
        else:
            shape = (obs_space.shape[0],)
        dtype = obs_space.dtype
        ret = [SpaceInfo([], is_img, shape, dtype)]
    elif isinstance(obs_space, gym.spaces.Dict):
        ret = []
        for sub_obs_key, sub_obs_space in obs_space.spaces.items():
            sub_info = _get_space_info(sub_obs_space)
            for si in sub_info:
                ret.append(SpaceInfo([sub_obs_key]+si.indexes, si.is_img, si.shape, si.dtype))
    else:
        raise UnexpectedException(f"Unexpected obs_space {obs_space}")

    return ret

def get_space_info(obs_space):
    ret = _get_space_info(obs_space)
    keys = [":".join(si.indexes) for si in ret]
    shapes = {":".join(si.indexes) : si.shape for si in ret}
    dtypes = {":".join(si.indexes) : si.dtype for si in ret}
    for i in range(len(keys)):
        if keys[i] == ():
            keys[i] = None
    return keys, shapes, dtypes


def get_sub_obs(obs, indexes):
    for i in indexes.split(":"):
        obs = obs[i]
    return obs