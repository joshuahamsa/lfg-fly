"""The fly's voice: naming a look with no API (spec §3.2) and the daily post (§4.5).

`nearest_name` labels the day's look with the critic's read of the nearest judged
look; `before_after` draws the link-free 1200×675 card from the hero's on-chain
images; `compose` writes the post text and `publish` puts it out, to X when an X
poster is wired in and to the outbox otherwise. Nothing here imports the body at
module import time.
"""

from lfg_fly.voice.card import before_after, change_lines, with_from
from lfg_fly.voice.naming import load_critic_records, nearest, nearest_name, similarity
from lfg_fly.voice.post import compose, day_number, publish, x_credentials, x_length

__all__ = [
    "before_after", "change_lines", "compose", "day_number", "load_critic_records",
    "nearest", "nearest_name", "publish", "similarity", "with_from", "x_credentials",
    "x_length",
]
