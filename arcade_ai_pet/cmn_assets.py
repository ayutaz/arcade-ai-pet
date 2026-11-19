"""Explicit CMN numeric format; width alone does not identify a frontend."""
from . import decision_assets

RUNTIME = 'context-duration-cmn-class-v1'
CONTRACT = dict(decision_assets.CONTRACT, frontend='context-duration-cmn',
                cmn='post-bin-mean-20-clip16')


def export_assets(weights, table, output):
    return decision_assets._export_assets(weights, table, output, RUNTIME, CONTRACT)
