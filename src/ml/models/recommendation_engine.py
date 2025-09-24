#!/usr/bin/env python3
"""
Enhanced Recommendation Engine for Educational Interactions

Generates specific, research-backed recommendations for improving educator-child
interactions with priority ranking, citations, and actionable next steps.

Author: Claude-4 (Partner-Level Microsoft SDE)
Issue: #51 - Story 2.5: Get actionable improvement recommendations
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
from enum import Enum


class RecommendationPriority(Enum):
    """Priority levels for recommendations"""
    CRITICAL = "critical"      # Immediate attention needed
    HIGH = "high"             # Major impact on learning
    MEDIUM = "medium"         # Moderate improvement opportunity
    LOW = "low"              # Fine-tuning suggestions


class RecommendationCategory(Enum):
    """Categories of educational recommendations"""
    QUESTIONING = "questioning"
    WAIT_TIME = "wait_time"
    EMOTIONAL_SUPPORT = "emotional_support"
    SCAFFOLDING = "scaffolding"
    CLASSROOM_ORGANIZATION = "classroom_organization"
    LANGUAGE_MODELING = "language_modeling"
    ENGAGEMENT = "engagement"


@dataclass
class ResearchCitation:
    """Research citation for recommendations"""
    authors: str
    year: int
    title: str
    journal: Optional[str] = None
    key_finding: str = ""

    def format_citation(self) -> str:
        """Format citation in academic style"""
        if self.journal:
            return f"{self.authors} ({self.year}). {self.title}. {self.journal}."
        return f"{self.authors} ({self.year}). {self.title}."


@dataclass
class ActionableRecommendation:
    """Enhanced recommendation with research backing and examples"""
    title: str
    description: str
    priority: RecommendationPriority
    category: RecommendationCategory

    # Research backing
    research_citations: List[ResearchCitation]
    evidence_strength: float  # 0.0 to 1.0

    # Actionable guidance
    specific_actions: List[str]
    before_example: Optional[str] = None
    after_example: Optional[str] = None

    # Context
    rationale: str = ""
    expected_impact: str = ""
    implementation_time: str = ""  # "immediate", "1-2 sessions", "ongoing"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "category": self.category.value,
            "research_citations": [
                {
                    "citation": cite.format_citation(),
                    "key_finding": cite.key_finding
                } for cite in self.research_citations
            ],
            "evidence_strength": self.evidence_strength,
            "specific_actions": self.specific_actions,
            "before_example": self.before_example,
            "after_example": self.after_example,
            "rationale": self.rationale,
            "expected_impact": self.expected_impact,
            "implementation_time": self.implementation_time
        }


class EnhancedRecommendationEngine:
    """
    Research-backed recommendation generation for educational interactions
    """

    def __init__(self):
        self.research_database = self._initialize_research_database()

    def _initialize_research_database(self) -> Dict[str, List[ResearchCitation]]:
        """Initialize database of educational research citations"""
        return {
            "questioning": [
                ResearchCitation(
                    "Hart, B., & Risley, T. R.",
                    1995,
                    "Meaningful Differences in the Lives of American Children",
                    key_finding="Quality of adult-child conversation more important than quantity for language development"
                ),
                ResearchCitation(
                    "Dickinson, D. K., & Porche, M. V.",
                    2011,
                    "Relation Between Language Experiences in Preschool and Later Reading Comprehension",
                    "Journal of Educational Psychology",
                    key_finding="Open-ended questions predict stronger reading comprehension outcomes"
                ),
                ResearchCitation(
                    "Rowe, M. L.",
                    2012,
                    "A longitudinal investigation of the role of quality and quantity in language input",
                    "Child Development",
                    key_finding="Higher quality questions associated with faster vocabulary growth"
                )
            ],
            "wait_time": [
                ResearchCitation(
                    "Rowe, M. B.",
                    1986,
                    "Wait Time: Slowing Down May Be A Way of Speeding Up!",
                    "Journal of Teacher Education",
                    key_finding="3-5 second wait time increases response length and complexity"
                ),
                ResearchCitation(
                    "Stahl, R. J.",
                    1994,
                    "Using Think-Time and Wait-Time Skillfully in the Classroom",
                    key_finding="Extended wait time improves student thinking and reduces anxiety"
                )
            ],
            "emotional_support": [
                ResearchCitation(
                    "Pianta, R. C., La Paro, K. M., & Hamre, B. K.",
                    2008,
                    "Classroom Assessment Scoring System (CLASS) Manual",
                    key_finding="Emotional support predicts academic and social development outcomes"
                ),
                ResearchCitation(
                    "Burchinal, M., Vandergrift, N., Pianta, R., & Mashburn, A.",
                    2010,
                    "Threshold analysis of association between child care quality and child outcomes",
                    "Early Childhood Research Quarterly",
                    key_finding="Emotional support quality has threshold effects on child development"
                )
            ],
            "scaffolding": [
                ResearchCitation(
                    "Vygotsky, L. S.",
                    1978,
                    "Mind in Society: The Development of Higher Psychological Processes",
                    key_finding="Learning occurs in Zone of Proximal Development through guided interaction"
                ),
                ResearchCitation(
                    "Wood, D., Bruner, J. S., & Ross, G.",
                    1976,
                    "The role of tutoring in problem solving",
                    "Journal of Child Psychology and Psychiatry",
                    key_finding="Effective scaffolding provides just enough support for independent achievement"
                )
            ]
        }

    def generate_recommendations(
        self,
        ml_predictions: Dict[str, Any],
        class_scores: Dict[str, float],
        scaffolding_analysis: Optional[Dict[str, Any]] = None,
        transcript_analysis: Optional[Dict[str, Any]] = None
    ) -> List[ActionableRecommendation]:
        """
        Generate comprehensive, research-backed recommendations

        Args:
            ml_predictions: ML model predictions
            class_scores: CLASS framework scores
            scaffolding_analysis: Scaffolding and ZPD analysis results
            transcript_analysis: Additional transcript analysis

        Returns:
            List of prioritized, actionable recommendations
        """
        recommendations = []

        # Generate recommendations by category
        recommendations.extend(self._generate_questioning_recommendations(ml_predictions, transcript_analysis))
        recommendations.extend(self._generate_wait_time_recommendations(ml_predictions))
        recommendations.extend(self._generate_emotional_support_recommendations(class_scores))
        recommendations.extend(self._generate_scaffolding_recommendations(ml_predictions, scaffolding_analysis))
        recommendations.extend(self._generate_engagement_recommendations(ml_predictions, class_scores))

        # Sort by priority and evidence strength
        recommendations = self._prioritize_recommendations(recommendations)

        # Return top 5 recommendations
        return recommendations[:5]

    def _generate_questioning_recommendations(
        self,
        predictions: Dict[str, Any],
        transcript_analysis: Optional[Dict[str, Any]]
    ) -> List[ActionableRecommendation]:
        """Generate questioning strategy recommendations"""
        recommendations = []

        question_quality = predictions.get("question_quality", 0.5)
        open_ended_ratio = predictions.get("open_ended_questions", 0.5)

        if question_quality < 0.7:
            priority = RecommendationPriority.HIGH if question_quality < 0.5 else RecommendationPriority.MEDIUM

            recommendations.append(ActionableRecommendation(
                title="Increase Open-Ended Question Usage",
                description="Transform closed questions into open-ended inquiries that promote deeper thinking and language development.",
                priority=priority,
                category=RecommendationCategory.QUESTIONING,
                research_citations=self.research_database["questioning"][:2],
                evidence_strength=0.85,
                specific_actions=[
                    "Replace 'yes/no' questions with 'how' and 'why' questions",
                    "Use phrases like 'Tell me more about...' or 'What do you think would happen if...'",
                    "Ask follow-up questions that build on child responses",
                    "Pause to let children formulate complex thoughts"
                ],
                before_example="Teacher: 'Is the block red?' Child: 'Yes.'",
                after_example="Teacher: 'What do you notice about this block?' Child: 'It's red and it's really smooth and heavy!'",
                rationale="Open-ended questions activate higher-order thinking skills and encourage extended verbal responses, supporting both cognitive and language development.",
                expected_impact="Children will provide longer, more complex responses and engage in deeper thinking about concepts.",
                implementation_time="immediate"
            ))

        if open_ended_ratio < 0.6:
            recommendations.append(ActionableRecommendation(
                title="Build Conversational Chains",
                description="Create extended conversations by asking follow-up questions that build on children's responses.",
                priority=RecommendationPriority.MEDIUM,
                category=RecommendationCategory.QUESTIONING,
                research_citations=[self.research_database["questioning"][2]],
                evidence_strength=0.78,
                specific_actions=[
                    "Listen carefully to child's initial response",
                    "Ask 'What else?' or 'Can you tell me more?'",
                    "Reference specific details from child's answer",
                    "Connect to child's interests and experiences"
                ],
                before_example="Child: 'I built a castle.' Teacher: 'Nice job.'",
                after_example="Child: 'I built a castle.' Teacher: 'What materials did you use for your castle? How did you make it so tall?'",
                rationale="Extended conversational exchanges provide more opportunities for vocabulary development and complex thinking.",
                expected_impact="Richer vocabulary usage and improved narrative skills in children.",
                implementation_time="1-2 sessions"
            ))

        return recommendations

    def _generate_wait_time_recommendations(self, predictions: Dict[str, Any]) -> List[ActionableRecommendation]:
        """Generate wait time recommendations"""
        recommendations = []

        wait_time_score = predictions.get("wait_time_appropriate", 0.7)

        if wait_time_score < 0.75:
            priority = RecommendationPriority.HIGH if wait_time_score < 0.5 else RecommendationPriority.MEDIUM

            recommendations.append(ActionableRecommendation(
                title="Implement Strategic Wait Time",
                description="Allow 3-5 seconds of silence after asking questions to give children time to process and formulate thoughtful responses.",
                priority=priority,
                category=RecommendationCategory.WAIT_TIME,
                research_citations=self.research_database["wait_time"],
                evidence_strength=0.82,
                specific_actions=[
                    "Count slowly to 5 after asking a question before speaking",
                    "Use non-verbal encouragement (nods, smiles) during wait time",
                    "Resist the urge to rephrase or repeat questions immediately",
                    "Allow comfortable silence for thinking"
                ],
                before_example="Teacher: 'Why do you think that happened?' [0.5 second pause] 'Well, maybe it's because...'",
                after_example="Teacher: 'Why do you think that happened?' [5 second pause] Child: 'I think it's because the water was too hot and it made the ice melt really fast.'",
                rationale="Extended wait time allows children to process complex questions and formulate more sophisticated responses.",
                expected_impact="Children will provide longer, more thoughtful answers and show increased confidence in responding.",
                implementation_time="immediate"
            ))

        return recommendations

    def _generate_emotional_support_recommendations(self, class_scores: Dict[str, float]) -> List[ActionableRecommendation]:
        """Generate emotional support recommendations based on CLASS scores"""
        recommendations = []

        emotional_support = class_scores.get("emotional_support", 4.0)

        if emotional_support < 4.0:
            priority = RecommendationPriority.HIGH if emotional_support < 3.0 else RecommendationPriority.MEDIUM

            recommendations.append(ActionableRecommendation(
                title="Enhance Emotional Validation and Support",
                description="Increase acknowledgment of children's feelings and provide more emotional support during interactions.",
                priority=priority,
                category=RecommendationCategory.EMOTIONAL_SUPPORT,
                research_citations=self.research_database["emotional_support"],
                evidence_strength=0.88,
                specific_actions=[
                    "Name and validate children's emotions: 'I can see you're excited about that!'",
                    "Use warm, responsive tone and body language",
                    "Comfort children when they're frustrated or upset",
                    "Celebrate children's efforts and progress, not just outcomes"
                ],
                before_example="Child seems frustrated. Teacher continues with lesson.",
                after_example="Child seems frustrated. Teacher: 'I notice you look frustrated. Building with these blocks can be tricky sometimes. What would help you feel better?'",
                rationale="Emotional support creates a secure base for learning and helps children develop emotional regulation skills.",
                expected_impact="Children will show increased engagement, willingness to take risks, and better emotional regulation.",
                implementation_time="ongoing"
            ))

        return recommendations

    def _generate_scaffolding_recommendations(
        self,
        predictions: Dict[str, Any],
        scaffolding_analysis: Optional[Dict[str, Any]]
    ) -> List[ActionableRecommendation]:
        """Generate scaffolding recommendations"""
        recommendations = []

        scaffolding_score = predictions.get("scaffolding_present", 0.6)

        if scaffolding_score < 0.7:
            recommendations.append(ActionableRecommendation(
                title="Implement Graduated Scaffolding Techniques",
                description="Provide strategic support that helps children reach their learning goals while building independence.",
                priority=RecommendationPriority.MEDIUM,
                category=RecommendationCategory.SCAFFOLDING,
                research_citations=self.research_database["scaffolding"],
                evidence_strength=0.80,
                specific_actions=[
                    "Start with minimal hints and gradually increase support if needed",
                    "Model thinking processes out loud",
                    "Break complex tasks into smaller, manageable steps",
                    "Fade support as children demonstrate competence"
                ],
                before_example="Child struggles with puzzle. Teacher: 'Let me show you how to do it.'",
                after_example="Child struggles with puzzle. Teacher: 'What piece do you think might fit here? Look at the shapes and colors.'",
                rationale="Appropriate scaffolding supports learning in the Zone of Proximal Development without creating dependence.",
                expected_impact="Children will develop problem-solving skills and increased confidence in tackling challenges.",
                implementation_time="ongoing"
            ))

        return recommendations

    def _generate_engagement_recommendations(
        self,
        predictions: Dict[str, Any],
        class_scores: Dict[str, float]
    ) -> List[ActionableRecommendation]:
        """Generate engagement-focused recommendations"""
        recommendations = []

        # Check for low engagement indicators
        overall_quality = (
            predictions.get("question_quality", 0.5) +
            predictions.get("scaffolding_present", 0.5) +
            class_scores.get("emotional_support", 4.0) / 7.0
        ) / 3

        if overall_quality < 0.6:
            recommendations.append(ActionableRecommendation(
                title="Increase Child-Centered Interaction Strategies",
                description="Follow children's interests more closely and build conversations around their natural curiosity.",
                priority=RecommendationPriority.MEDIUM,
                category=RecommendationCategory.ENGAGEMENT,
                research_citations=[self.research_database["questioning"][0]],
                evidence_strength=0.75,
                specific_actions=[
                    "Observe what captures children's attention naturally",
                    "Ask questions about their interests and observations",
                    "Connect learning opportunities to their fascinations",
                    "Allow children to lead parts of the conversation"
                ],
                before_example="Teacher follows planned lesson despite child's interest in butterfly outside window.",
                after_example="Teacher: 'I noticed you're watching that butterfly! What do you see? How do you think it flies?'",
                rationale="Child-centered approaches increase motivation and create more meaningful learning experiences.",
                expected_impact="Higher engagement levels and more enthusiastic participation in learning activities.",
                implementation_time="immediate"
            ))

        return recommendations

    def _prioritize_recommendations(self, recommendations: List[ActionableRecommendation]) -> List[ActionableRecommendation]:
        """Sort recommendations by priority and evidence strength"""
        priority_weights = {
            RecommendationPriority.CRITICAL: 4,
            RecommendationPriority.HIGH: 3,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 1
        }

        def sort_key(rec: ActionableRecommendation) -> Tuple[int, float]:
            return (priority_weights[rec.priority], rec.evidence_strength)

        return sorted(recommendations, key=sort_key, reverse=True)


# Factory function for integration with existing API
def create_recommendation_engine():
    """Factory function to create recommendation engine instance"""
    return EnhancedRecommendationEngine()