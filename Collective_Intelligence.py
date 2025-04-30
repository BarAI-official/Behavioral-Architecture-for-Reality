"""
BarAI Collective Intelligence Module

Aggregates emotional and engagement metrics across user groups,
offering mood distribution, trend detection, and intervention suggestions.

Authors: BarAI Dev Team
"""
import logging
from statistics import mean, pstdev
from typing import Dict, List, Any

from .logging_config import get_logger
from .config import Config

logger = get_logger(__name__)

class CollectiveIntelligence:
    """
    Provides tools to assess group emotional states and recommend interventions.
    """
    def __init__(self, config: Config, sentiment_scores: List[float]):
        self.config = config
        self.scores = sentiment_scores
        logger.info("CollectiveIntelligence initialized with %d samples", len(sentiment_scores))

    def mood_statistics(self) -> Dict[str, Any]:
        """
        Compute statistical distribution of sentiment scores.
        Returns mean, standard deviation, count, min, max.
        """
        if not self.scores:
            logger.warning("No sentiment scores provided.")
            return {}
        stats = {
            'mean': round(mean(self.scores), 3),
            'std_dev': round(pstdev(self.scores), 3),
            'count': len(self.scores),
            'min': round(min(self.scores), 3),
            'max': round(max(self.scores), 3)
        }
        logger.debug(f"Mood stats: {stats}")
        return stats

    def intervention_suggestions(self) -> List[str]:
        """
        Suggest actions based on group mood and engagement trends.
        """
        stats = self.mood_statistics()
        suggestions = []
        mean_score = stats.get('mean', 0)
        if mean_score < 0.4:
            suggestions.append('Schedule a 10-minute mindfulness break')
        if mean_score > 0.7:
            suggestions.append('Recognize high performers in group chat')
        # Feature flag: beta plugin suggestion
        if self.config.features.enable_beta_plugins:
            suggestions.append('Enable Beta Emotion Analytics Plugin')
        logger.info(f"Intervention suggestions: {suggestions}")
        return suggestions

# Example usage
def demo_collective_intelligence():
    cfg = Config()
    ci = CollectiveIntelligence(cfg, [0.2, 0.5, 0.8, 0.9])
    print(ci.mood_statistics())
    print(ci.intervention_suggestions())

if __name__ == '__main__':
    demo_collective_intelligence()
