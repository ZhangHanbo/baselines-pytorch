# TODO: add the following commented environments into the register
# from BaxterReacherv0 import *
# from myenvs.robosuite.robosuite import *

import copy
from .registration import register, make, registry, spec


register(
    id='FetchThrowDice-v0',
    entry_point='myenvs.fetch:FetchThrowDiceEnv',
    kwargs={},
    max_episode_steps=50,
)

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type':reward_type,
    }

    for i in range(2, 101):
        _kwargs = copy.deepcopy(kwargs)
        _kwargs["n_bits"] = i
        register(
            id='FlipBit{}{:d}-v0'.format(suffix, i),
            entry_point='myenvs.toy:FlipBit',
            kwargs=_kwargs,
            max_episode_steps = i,
        )

    for i in range(2, 51):
        _kwargs = copy.deepcopy(kwargs)
        _kwargs["layout"] = (i, i)
        _kwargs["max_steps"] = 2 * i - 2
        register(
            id='EmptyMaze{}{:d}-v0'.format(suffix, i),
            entry_point='myenvs.toy:EmptyMaze',
            kwargs=_kwargs,
            max_episode_steps=_kwargs["max_steps"],
        )

    register(
        id='FourRoom{}-v0'.format(suffix),
        entry_point='myenvs.toy:FourRoomMaze',
        kwargs=kwargs,
        max_episode_steps=32,
    )

    register(
        id='FetchReachDiscrete{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchReachDiscrete',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPushDiscrete{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchPushDiscrete',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchSlideDiscrete{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchSlideDiscrete',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchThrow{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchThrowEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchThrowRubberBall{}-v0'.format(suffix),
        entry_point='myenvs.fetch:FetchThrowRubberBallEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='DragRope{}-v0'.format(suffix),
        entry_point='myenvs.ravens:DragRopeEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='SweepPile{}-v0'.format(suffix),
        entry_point='myenvs.ravens:SweepPileEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='MsPacman{}-v0'.format(suffix),
        entry_point='myenvs.atari.mspacman:MsPacman',
        kwargs=kwargs,
        max_episode_steps=26,
    )

    _rope_kwargs = {
        'observation_mode': 'key_point',
        'action_mode': 'picker',
        'num_picker': 2,
        'render': True,
        'headless': True,
        'horizon': 50,
        'action_repeat': 8,
        'render_mode': 'cloth',
        'num_variations': 1,
        'use_cached_states': False,
        'deterministic': False,
        'save_cached_states': False,
    }
    _rope_kwargs.update(kwargs)
    register(
        id='RopeConfiguration{}-v0'.format(suffix),
        entry_point='myenvs.softgym:RopeConfigurationEnv',
        kwargs=_rope_kwargs,
        max_episode_steps=50,
    )
