from typing import Sequence
from frozendict import frozendict

from pyfemtet.opt.history import History, TrialState
from pyfemtet.opt.problem.variable_manager import SupportedVariableTypes


__all__ = [
    'TrialQueue',
    'get_tried_list_from_history',
    '_IS_INITIAL_TRIAL_FLAG_KEY',
]


_IS_INITIAL_TRIAL_FLAG_KEY = 'is_initial_trial'


class EnqueuedTrial:

    def __init__(self, d: dict[str, SupportedVariableTypes], flags: dict = None):
        self.d = d
        self.flags = flags or {}

    @staticmethod
    def d_type_verified(d) -> dict:
        out = {}
        for k, v in d.items():
            if isinstance(v, str):
                out.update({k: v})
            else:
                out.update({k: float(v)})
        return out

    def get_hashed_id(self):
        return hash(frozendict(self.d_type_verified(self.d)))


class TrialQueue:

    def __init__(self):
        self.queue: list[EnqueuedTrial] = []

    def __len__(self):
        return len(self.queue)

    def enqueue_first(self, d: dict[str, SupportedVariableTypes], flags: dict = None):
        self.queue.insert(0, EnqueuedTrial(d, flags=flags))

    def enqueue(self, d: dict[str, SupportedVariableTypes], flags: dict = None):
        self.queue.append(EnqueuedTrial(d, flags=flags))

    def dequeue(self) -> dict[str, SupportedVariableTypes] | None:
        if len(self.queue) == 0:
            return None
        return self.queue.pop(0).d

    def remove_duplicated(self):
        indices_to_remove = []
        all_ids = [t.get_hashed_id() for t in self.queue]

        # 先に queue に入れられたものから
        # 順に見ていく
        # indices は sorted となるはず
        seen_ids = set()
        for i, id in enumerate(all_ids):
            if id in seen_ids:
                indices_to_remove.append(i)
            else:
                seen_ids.add(id)

        # 削除
        for i in indices_to_remove[::-1]:
            self.queue.pop(i)

    def remove_tried(self, tried_list: Sequence[dict[str, SupportedVariableTypes]]):
        indices_to_remove = []
        tried_id = [EnqueuedTrial(tried).get_hashed_id() for tried in tried_list]
        for i, t in enumerate(self.queue):
            if t.get_hashed_id() in tried_id:
                indices_to_remove.append(i)
        for i in indices_to_remove[::-1]:
            self.queue.pop(i)


def get_tried_list_from_history(
        history: History,
        equality_filters=None,
) -> list[dict[str, SupportedVariableTypes]]:
    out = []
    df = history.get_df(equality_filters=equality_filters)
    for _, row in df.iterrows():
        # Want to retry if unknown error
        # so don't count it as tried.
        if row['state'] in (
                TrialState.unknown_error,
                TrialState.undefined,
        ):
            continue
        d = row[history.prm_names].to_dict()
        out.append(d)
    return out
