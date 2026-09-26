"""A deliberately label-free reward for pure OPD.

VERL's PPO entrypoint requires a reward callback even when
``use_task_rewards=False``.  Returning zero here keeps the task-reward path
well-defined while ensuring that ranking labels and output format do not
participate in the update.
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    del data_source, solution_str, ground_truth, extra_info
    return 0.0
