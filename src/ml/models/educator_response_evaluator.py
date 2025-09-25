"""
Educator Response Evaluator

ML model for evaluating educator responses across pedagogical categories.
Placeholder implementation for MVP Sprint 1 - will be replaced with trained models.

Author: Claude (Partner-Level Microsoft SDE)
Issue: #108 - PIVOT to User Response Evaluation
"""

import asyncio
import re
from typing import Dict, Any, List
import random

class EducatorResponseEvaluator:
    """
    Evaluates educator responses across evidence-based pedagogical categories.

    Current implementation: Rule-based heuristics
    Future: Trained ML models from Issue #109
    """

    def __init__(self):
        self.category_evaluators = {
            "emotional_support": self._evaluate_emotional_support,
            "scaffolding_support": self._evaluate_scaffolding_support,
            "language_quality": self._evaluate_language_quality,
            "developmental_appropriateness": self._evaluate_developmental_appropriateness,
            "overall_effectiveness": self._evaluate_overall_effectiveness
        }

    async def evaluate_category(self,
                               educator_response: str,
                               scenario_context: str,
                               category_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate educator response for a specific category.

        Args:
            educator_response: User's typed response
            scenario_context: Background scenario information
            category_definition: Category details and criteria

        Returns:
            Dict with score, feedback, strengths, and growth areas
        """
        category_id = category_definition["id"]

        # Simulate processing time
        await asyncio.sleep(0.5)

        if category_id in self.category_evaluators:
            return await self.category_evaluators[category_id](educator_response, scenario_context)
        else:
            return await self._evaluate_generic_category(educator_response, category_definition)

    async def _evaluate_emotional_support(self, response: str, context: str) -> Dict[str, Any]:
        """Evaluate emotional support and validation."""
        strengths = []
        growth_areas = []
        score = 5.0  # Default middle score

        # Check for emotion acknowledgment
        emotion_keywords = ["frustrated", "upset", "feeling", "see you", "understand", "okay", "that's hard"]
        if any(keyword in response.lower() for keyword in emotion_keywords):
            strengths.append("You acknowledged Maya's emotional state")
            score += 1.5

        # Check for validation
        validation_phrases = ["that's okay", "it's alright", "I understand", "that happens"]
        if any(phrase in response.lower() for phrase in validation_phrases):
            strengths.append("You provided emotional validation")
            score += 1.0

        # Check for dismissive language
        dismissive_words = ["don't", "stop", "shouldn't", "just try"]
        if any(word in response.lower() for word in dismissive_words):
            growth_areas.append("Consider avoiding dismissive language that minimizes feelings")
            score -= 1.0

        # Growth opportunity if no emotion labeling
        if not any(keyword in response.lower() for keyword in ["frustrated", "upset", "mad", "angry"]):
            growth_areas.append("Try labeling emotions explicitly: 'I can see you're feeling frustrated'")

        return {
            "score": min(10.0, max(0.0, score)),
            "feedback": "Assessment of emotional support and validation techniques",
            "strengths": strengths,
            "growth_areas": growth_areas,
            "evidence_alignment": 0.8 if strengths else 0.4
        }

    async def _evaluate_scaffolding_support(self, response: str, context: str) -> Dict[str, Any]:
        """Evaluate scaffolding and learning support."""
        strengths = []
        growth_areas = []
        score = 5.0

        # Check for breaking down tasks
        scaffolding_indicators = ["one piece", "small step", "try this", "let's start with", "what if we"]
        if any(indicator in response.lower() for indicator in scaffolding_indicators):
            strengths.append("You provided appropriate task scaffolding")
            score += 1.5

        # Check for building on success
        success_building = ["you did", "I noticed", "you got", "good job on"]
        if any(phrase in response.lower() for phrase in success_building):
            strengths.append("You built on what the child accomplished")
            score += 1.0

        # Check for offering help
        help_offers = ["would you like", "should we", "do you want", "let me help"]
        if any(offer in response.lower() for offer in help_offers):
            strengths.append("You offered appropriate assistance")
            score += 0.5

        if len(strengths) == 0:
            growth_areas.append("Consider breaking the task into smaller, manageable steps")
            growth_areas.append("Try building on what Maya has already accomplished")

        return {
            "score": min(10.0, max(0.0, score)),
            "feedback": "Assessment of scaffolding and learning support strategies",
            "strengths": strengths,
            "growth_areas": growth_areas,
            "evidence_alignment": 0.75 if len(strengths) >= 2 else 0.5
        }

    async def _evaluate_language_quality(self, response: str, context: str) -> Dict[str, Any]:
        """Evaluate language and communication quality."""
        strengths = []
        growth_areas = []
        score = 5.0

        # Check for age-appropriate language
        word_count = len(response.split())
        if 15 <= word_count <= 50:  # Appropriate length for 4-year-old
            strengths.append("Response length is developmentally appropriate")
            score += 1.0

        # Check for positive language
        positive_words = ["can", "will", "good", "great", "wonderful", "nice"]
        positive_count = sum(1 for word in positive_words if word in response.lower())
        if positive_count >= 2:
            strengths.append("You used positive, encouraging language")
            score += 1.0

        # Check for complex language that might confuse
        complex_words = ["frustrated", "disappointed", "accomplish", "challenge"]
        if any(word in response.lower() for word in complex_words):
            growth_areas.append("Consider simplifying language for a 4-year-old")
            score -= 0.5

        # Check for questions to engage
        question_count = response.count('?')
        if question_count >= 1:
            strengths.append("You engaged Maya with questions")
            score += 0.5

        return {
            "score": min(10.0, max(0.0, score)),
            "feedback": "Assessment of language and communication appropriateness",
            "strengths": strengths,
            "growth_areas": growth_areas,
            "evidence_alignment": 0.7 if strengths else 0.4
        }

    async def _evaluate_developmental_appropriateness(self, response: str, context: str) -> Dict[str, Any]:
        """Evaluate developmental appropriateness."""
        strengths = []
        growth_areas = []
        score = 6.0  # Slightly higher baseline

        # Check for age-appropriate expectations
        if "try again" in response.lower() and "choice" in response.lower():
            strengths.append("You offered age-appropriate choices")
            score += 1.0

        # Check for patience indicators
        patience_indicators = ["take time", "when you're ready", "break", "come back"]
        if any(indicator in response.lower() for indicator in patience_indicators):
            strengths.append("You showed patience appropriate for a 4-year-old's attention span")
            score += 1.0

        # Check for unrealistic expectations
        pressure_words = ["finish", "must", "should", "have to"]
        if any(word in response.lower() for word in pressure_words):
            growth_areas.append("Avoid putting pressure on preschoolers to complete difficult tasks")
            score -= 1.0

        return {
            "score": min(10.0, max(0.0, score)),
            "feedback": "Assessment of developmental appropriateness",
            "strengths": strengths,
            "growth_areas": growth_areas,
            "evidence_alignment": 0.8 if strengths else 0.5
        }

    async def _evaluate_overall_effectiveness(self, response: str, context: str) -> Dict[str, Any]:
        """Evaluate overall effectiveness of response."""
        strengths = []
        growth_areas = []

        # Holistic assessment based on multiple factors
        factors_score = 0

        # Factor 1: Addresses the core issue (frustration)
        if any(word in response.lower() for word in ["frustrated", "upset", "hard", "tricky"]):
            factors_score += 2
            strengths.append("You addressed Maya's core emotional state")

        # Factor 2: Provides concrete next steps
        if any(phrase in response.lower() for phrase in ["try", "do", "would you like", "let's"]):
            factors_score += 2
            strengths.append("You provided actionable next steps")

        # Factor 3: Maintains relationship
        relationship_words = ["maya", "with you", "together", "help"]
        if any(word in response.lower() for word in relationship_words):
            factors_score += 1
            strengths.append("You maintained connection with Maya")

        # Factor 4: Offers autonomy
        if "?" in response or any(phrase in response.lower() for phrase in ["would you", "do you want"]):
            factors_score += 1
            strengths.append("You respected Maya's autonomy with choices")

        score = min(10.0, factors_score + 2)  # Base score of 2

        if len(strengths) < 2:
            growth_areas.append("Consider combining emotional support with concrete assistance")

        return {
            "score": score,
            "feedback": "Holistic assessment of response effectiveness",
            "strengths": strengths,
            "growth_areas": growth_areas,
            "evidence_alignment": min(1.0, factors_score / 6.0)
        }

    async def _evaluate_generic_category(self, response: str, category_def: Dict[str, Any]) -> Dict[str, Any]:
        """Generic evaluation for unknown categories."""
        # Basic heuristic evaluation
        word_count = len(response.split())
        score = min(10.0, max(3.0, (word_count / 20) * 5 + random.uniform(2, 4)))

        return {
            "score": round(score, 1),
            "feedback": f"Assessment of {category_def['name']}",
            "strengths": ["Response shows engagement with the scenario"],
            "growth_areas": ["Consider more specific strategies for this category"],
            "evidence_alignment": 0.5
        }