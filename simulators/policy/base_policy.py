from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional


class BasePolicy:

    def __init__(self, id: str, config) -> None:
        super().__init__()
        self.id = id

    def report(self):
        print(self.id)
        if self.policy_observable is not None:
            print("  - The policy can observe:", end=' ')
            for i, k in enumerate(self.policy_observable):
                print(k, end='')
                if i == len(self.policy_observable) - 1:
                    print('.')
                else:
                    print(', ', end='')
        else:
            print("  - The policy can only access observation.")
