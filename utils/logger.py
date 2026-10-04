import csv
import os
from typing import List, Dict


class EpisodeLogger:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.episodes: List[Dict] = []
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def log(self, episode: int, reward: float, length: int, total_steps: int):
        entry = {
            "episode": episode,
            "reward": reward,
            "length": length,
            "total_steps": total_steps
        }
        self.episodes.append(entry)

    def save(self):
        with open(self.save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "reward", "length", "total_steps"])
            writer.writeheader()
            writer.writerows(self.episodes)

    def get_rewards(self) -> List[float]:
        return [e["reward"] for e in self.episodes]