"""行业轮动与机会发现工作流。

Usage:
    from sector_rotation import run
    report = run("最近哪个行业有机会？")
"""

from sector_rotation.pipeline import run

__all__ = ["run"]
