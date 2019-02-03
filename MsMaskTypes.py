from enum import Enum


class MsMaskTypes(Enum):
    EXPERT_1 = 0
    EXPERT_2 = 1
    INTERSECTION = 2
    UNION = 3
    EXPERT_1_DILATED = 4
    EXPERT_2_DILATED = 5
