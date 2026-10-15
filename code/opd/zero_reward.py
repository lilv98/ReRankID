# Copyright 2026 The ReRankID Authors
# SPDX-License-Identifier: Apache-2.0

"""Zero task reward for OPD (paper Sec. 5).

verl's PPO entry point requires a reward callback even with use_task_rewards=False. Returning zero
keeps that path defined while ranking labels and output format never enter the update.
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    del data_source, solution_str, ground_truth, extra_info
    return 0.0
