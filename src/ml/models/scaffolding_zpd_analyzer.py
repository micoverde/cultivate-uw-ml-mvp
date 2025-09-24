#!/usr/bin/env python3
"""
Scaffolding and Zone of Proximal Development (ZPD) Analysis
Enhanced analysis of teaching techniques and developmental appropriateness.

Author: Claude-4 (Partner-Level Microsoft SDE)
Issue: #49 - Story 2.3: Scaffolding Technique Identification
"""

from typing import Dict, Any, List, Tuple, Optional
import re
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ZPDIndicator:
    """Represents a Zone of Proximal Development indicator"""

    def __init__(self, indicator_type: str, evidence: str, confidence: float,
                 research_backing: str, line_numbers: List[int]):
        self.indicator_type = indicator_type
        self.evidence = evidence
        self.confidence = confidence
        self.research_backing = research_backing
        self.line_numbers = line_numbers

class ScaffoldingTechnique:
    """Represents an identified scaffolding technique"""

    def __init__(self, technique_type: str, description: str, evidence: List[str],
                 effectiveness_score: float, research_citations: List[str]):
        self.technique_type = technique_type
        self.description = description
        self.evidence = evidence
        self.effectiveness_score = effectiveness_score
        self.research_citations = research_citations

class ScaffoldingZPDAnalyzer:
    """
    Advanced scaffolding and ZPD analysis for educational interactions.

    Identifies sophisticated teaching techniques including:
    - Scaffolding patterns (supportive prompts, guided discovery)
    - ZPD indicators (appropriate challenge level, developmental matching)
    - Fading support techniques (gradual independence building)
    - Wait time behaviors (thinking time, processing support)
    """

    def __init__(self, model_type: str = 'classical'):
        self.model_type = model_type
        self.version = f"{model_type}_v1.0"

        # Zone of Proximal Development Indicators
        self.zpd_indicators = {
            'appropriate_challenge': {
                'patterns': [
                    # Challenge within reach
                    r'\bthat\'s\s+(getting\s+closer|almost|nearly\s+there)',
                    r'\byou\'re\s+(on\s+the\s+right\s+track|thinking\s+well)',
                    r'\btry\s+(thinking\s+about|considering|looking\s+at)',
                    r'\bwhat\s+if\s+you\s+(tried|thought\s+about|considered)',

                    # Building on prior knowledge
                    r'\bremember\s+when\s+we',
                    r'\blike\s+we\s+did\s+(yesterday|before|last\s+time)',
                    r'\byou\s+know\s+how\s+to',
                    r'\bthink\s+about\s+what\s+you\s+already\s+know',
                ],
                'research': 'Vygotsky (1978) - Zone of Proximal Development theory',
                'description': 'Tasks appropriately challenging for child\'s developmental level'
            },

            'guided_discovery': {
                'patterns': [
                    # Leading questions for discovery
                    r'\bwhat\s+do\s+you\s+(notice|observe|see\s+happening)',
                    r'\bwhat\s+might\s+(happen|occur)\s+if',
                    r'\bhow\s+do\s+you\s+think\s+this\s+works',
                    r'\bwhat\s+patterns\s+do\s+you\s+(see|notice)',

                    # Encouraging exploration
                    r'\blet\'s\s+(explore|investigate|find\s+out)',
                    r'\bwhat\s+would\s+you\s+like\s+to\s+(try|explore|test)',
                    r'\bhow\s+could\s+we\s+(figure\s+this\s+out|solve\s+this)',
                ],
                'research': 'Bruner (1961) - Discovery learning theory',
                'description': 'Guiding children to discover concepts independently'
            },

            'developmental_matching': {
                'patterns': [
                    # Age-appropriate language
                    r'\bin\s+simple\s+words',
                    r'\blet\s+me\s+show\s+you\s+with',
                    r'\blike\s+a\s+(story|game|adventure)',
                    r'\bimagine\s+if\s+you\s+were\s+a',

                    # Concrete to abstract progression
                    r'\bfirst\s+let\'s\s+(touch|feel|hold)',
                    r'\bnow\s+that\s+you\'ve\s+(seen|felt|tried)',
                    r'\bthink\s+about\s+what\s+that\s+reminds\s+you\s+of',
                ],
                'research': 'Piaget (1977) - Cognitive development stages',
                'description': 'Language and concepts matched to developmental stage'
            }
        }

        # Advanced Scaffolding Techniques
        self.scaffolding_techniques = {
            'modeling_thinking': {
                'patterns': [
                    # Think-aloud demonstrations
                    r'\bi\'m\s+thinking',
                    r'\blet\s+me\s+think\s+out\s+loud',
                    r'\bhmmm,\s+i\s+wonder',
                    r'\bwhen\s+i\s+see\s+this,\s+i\s+think',

                    # Metacognitive modeling
                    r'\bi\'m\s+asking\s+myself',
                    r'\bthat\s+makes\s+me\s+wonder',
                    r'\bi\'m\s+noticing\s+that',
                    r'\bthis\s+reminds\s+me\s+of',
                ],
                'research': 'Wood, Bruner & Ross (1976) - Scaffolding concept',
                'description': 'Demonstrating thinking processes for children to internalize'
            },

            'graduated_prompting': {
                'patterns': [
                    # Least to most supportive prompts
                    r'\bwhat\s+do\s+you\s+think\?\s*.*\s*what\s+if',  # General then specific
                    r'\btry\s+again\s*.*\s*think\s+about',  # Encouragement then guidance
                    r'\bany\s+ideas\?\s*.*\s*remember\s+when',  # Open then connecting

                    # Sequential support levels
                    r'\bfirst.*then.*finally',
                    r'\bstart\s+with.*next.*after\s+that',
                    r'\btry\s+this\s+step.*then\s+this',
                ],
                'research': 'Pea (2004) - Graduated prompting in learning',
                'description': 'Providing increasingly specific support as needed'
            },

            'collaborative_construction': {
                'patterns': [
                    # Joint problem solving
                    r'\blet\'s\s+(figure\s+this\s+out|work\s+on\s+this)\s+together',
                    r'\bwhat\s+if\s+we\s+(both|together)',
                    r'\byou\s+start\s+and\s+i\'ll\s+help',
                    r'\bi\'ll\s+do\s+this\s+part,\s+you\s+do',

                    # Shared thinking
                    r'\bour\s+(idea|plan|discovery)',
                    r'\bwe\s+(figured\s+out|discovered|learned)',
                    r'\btogether\s+we',
                ],
                'research': 'Rogoff (1990) - Guided participation theory',
                'description': 'Adult and child working together as equal partners'
            },

            'fading_support': {
                'patterns': [
                    # Gradual release of support
                    r'\bnow\s+you\s+try\s+(by\s+yourself|on\s+your\s+own)',
                    r'\bdo\s+you\s+think\s+you\s+can\s+(handle|manage)\s+this',
                    r'\byou\'ve\s+got\s+this',
                    r'\bi\s+think\s+you\'re\s+ready\s+to',

                    # Independence encouragement
                    r'\bwhat\s+would\s+you\s+do\s+if\s+i\s+wasn\'t\s+here',
                    r'\byou\s+don\'t\s+need\s+my\s+help\s+anymore',
                    r'\bshow\s+me\s+what\s+you\s+learned',
                    r'\bi\s+bet\s+you\s+can\s+do\s+this\s+yourself',
                ],
                'research': 'Pearson & Gallagher (1983) - Gradual release model',
                'description': 'Systematically reducing support to build independence'
            }
        }

        # Wait Time Behavioral Patterns (enhanced from Issue #47)
        self.wait_time_behaviors = {
            'optimal_wait_time': {
                'patterns': [
                    # Teacher wait indicators
                    r'Teacher:\s*[^?]*\?\s*\n\s*Child:\s*(um+|hmm+|let\s+me\s+think)',
                    r'\?\s*\.\.\.\s*Child:',  # Ellipses indicating pause

                    # Child processing time
                    r'Child:\s*(um+|uh+|well).*\n\s*Teacher:\s*(that\'s|good|right)',
                    r'Child:\s*.*\.\.\.\s*(I\s+think|Maybe|It\s+might)',
                ],
                'research': 'Rowe (1986) - Wait time research in education',
                'description': 'Appropriate 3-5 second thinking time after questions'
            },

            'responsive_wait_time': {
                'patterns': [
                    # Adjusting wait time based on child needs
                    r'Teacher:\s*Take\s+your\s+time',
                    r'No\s+rush',
                    r'Think\s+about\s+it\s+for\s+a\s+moment',
                    r'I\s+can\s+wait',

                    # Recognizing processing needs
                    r'I\s+can\s+see\s+you\'re\s+thinking',
                    r'You\'re\s+working\s+hard\s+on\s+this',
                    r'That\'s\s+a\s+big\s+question',
                ],
                'research': 'Tobin (1987) - Differential wait time for different learners',
                'description': 'Adjusting wait time based on individual child needs'
            }
        }

    async def analyze(self, transcript: str) -> Dict[str, Any]:
        """Main analysis entry point for compatibility with ML pipeline"""
        return await self.analyze_scaffolding_zpd(transcript)

    async def analyze_scaffolding_zpd(self, transcript: str) -> Dict[str, Any]:
        """
        Comprehensive scaffolding and ZPD analysis of educational transcript.

        Returns detailed analysis including:
        - ZPD indicators with research backing
        - Scaffolding techniques with effectiveness scores
        - Wait time behaviors with appropriateness assessment
        - Fading support evidence with developmental progression
        """
        try:
            logger.debug("Starting comprehensive scaffolding and ZPD analysis")

            # Analyze ZPD indicators
            zpd_analysis = self._analyze_zpd_indicators(transcript)

            # Analyze scaffolding techniques
            scaffolding_analysis = self._analyze_scaffolding_techniques(transcript)

            # Enhanced wait time analysis
            wait_time_analysis = self._analyze_enhanced_wait_time(transcript)

            # Fading support analysis
            fading_analysis = self._analyze_fading_support(transcript)

            # Generate comprehensive report
            return {
                'zpd_indicators': zpd_analysis,
                'scaffolding_techniques': scaffolding_analysis,
                'wait_time_behaviors': wait_time_analysis,
                'fading_support': fading_analysis,
                'overall_assessment': self._generate_overall_assessment(
                    zpd_analysis, scaffolding_analysis, wait_time_analysis, fading_analysis
                ),
                'research_citations': self._compile_research_citations(),
                'recommendations': self._generate_recommendations(
                    zpd_analysis, scaffolding_analysis
                )
            }

        except Exception as e:
            logger.error(f"Scaffolding ZPD analysis failed: {e}")
            return await self._fallback_analysis(transcript)

    def _analyze_zpd_indicators(self, transcript: str) -> Dict[str, Any]:
        """Analyze Zone of Proximal Development indicators"""
        lines = transcript.split('\n')
        zpd_findings = {}

        for zpd_type, zpd_data in self.zpd_indicators.items():
            indicators = []
            total_confidence = 0

            for i, line in enumerate(lines):
                for pattern in zpd_data['patterns']:
                    matches = re.finditer(pattern, line.lower(), re.IGNORECASE)
                    for match in matches:
                        confidence = self._calculate_context_confidence(line, match.group())

                        indicators.append({
                            'evidence': match.group(),
                            'context': line.strip(),
                            'line_number': i + 1,
                            'confidence': confidence,
                            'research_backing': zpd_data['research']
                        })
                        total_confidence += confidence

            zpd_findings[zpd_type] = {
                'indicators': indicators,
                'frequency': len(indicators),
                'average_confidence': total_confidence / max(1, len(indicators)),
                'description': zpd_data['description'],
                'research_backing': zpd_data['research']
            }

        return zpd_findings

    def _analyze_scaffolding_techniques(self, transcript: str) -> Dict[str, Any]:
        """Analyze scaffolding techniques with effectiveness scoring"""
        lines = transcript.split('\n')
        technique_findings = {}

        for technique_type, technique_data in self.scaffolding_techniques.items():
            techniques_found = []
            effectiveness_scores = []

            for i, line in enumerate(lines):
                for pattern in technique_data['patterns']:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Calculate effectiveness based on context
                        effectiveness = self._assess_technique_effectiveness(
                            line, lines[max(0, i-1):min(len(lines), i+2)], technique_type
                        )

                        techniques_found.append({
                            'evidence': line.strip(),
                            'line_number': i + 1,
                            'effectiveness_score': effectiveness,
                            'pattern_matched': pattern
                        })
                        effectiveness_scores.append(effectiveness)

            technique_findings[technique_type] = {
                'techniques_found': techniques_found,
                'frequency': len(techniques_found),
                'average_effectiveness': sum(effectiveness_scores) / max(1, len(effectiveness_scores)),
                'description': technique_data['description'],
                'research_citation': technique_data['research']
            }

        return technique_findings

    def _analyze_enhanced_wait_time(self, transcript: str) -> Dict[str, Any]:
        """Enhanced wait time analysis with behavioral patterns"""
        wait_time_findings = {}

        for behavior_type, behavior_data in self.wait_time_behaviors.items():
            behaviors_found = []

            for pattern in behavior_data['patterns']:
                matches = re.finditer(pattern, transcript, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    behaviors_found.append({
                        'evidence': match.group(),
                        'start_pos': match.start(),
                        'end_pos': match.end(),
                        'quality_indicator': self._assess_wait_time_quality(match.group())
                    })

            wait_time_findings[behavior_type] = {
                'behaviors_found': behaviors_found,
                'frequency': len(behaviors_found),
                'description': behavior_data['description'],
                'research_backing': behavior_data['research']
            }

        return wait_time_findings

    def _analyze_fading_support(self, transcript: str) -> Dict[str, Any]:
        """Analyze evidence of fading support (gradual independence building)"""
        lines = transcript.split('\n')

        # Look for progression patterns indicating fading support
        support_levels = []
        independence_markers = []

        for i, line in enumerate(lines):
            # Detect support level in each teacher interaction
            if 'teacher:' in line.lower() or 'educator:' in line.lower():
                support_level = self._assess_support_level(line)
                support_levels.append({
                    'line_number': i + 1,
                    'support_level': support_level,
                    'evidence': line.strip()
                })

            # Detect independence encouragement
            independence_patterns = self.scaffolding_techniques['fading_support']['patterns']
            for pattern in independence_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    independence_markers.append({
                        'line_number': i + 1,
                        'evidence': line.strip(),
                        'independence_level': self._assess_independence_encouragement(line)
                    })

        # Analyze progression
        fading_progression = self._analyze_support_progression(support_levels)

        return {
            'support_levels_detected': support_levels,
            'independence_markers': independence_markers,
            'fading_progression': fading_progression,
            'overall_fading_score': fading_progression.get('progression_score', 0.5)
        }

    def _calculate_context_confidence(self, context: str, evidence: str) -> float:
        """Calculate confidence score based on surrounding context"""
        base_confidence = 0.7

        # Increase confidence for educational context indicators
        educational_indicators = ['child', 'student', 'learning', 'think', 'understand']
        for indicator in educational_indicators:
            if indicator in context.lower():
                base_confidence += 0.05

        # Increase confidence for question context
        if '?' in context:
            base_confidence += 0.1

        return min(1.0, base_confidence)

    def _assess_technique_effectiveness(self, line: str, context_lines: List[str],
                                      technique_type: str) -> float:
        """Assess effectiveness of scaffolding technique based on context"""
        effectiveness = 0.6  # Base score

        # Look for child response indicating engagement
        for context_line in context_lines:
            if 'child:' in context_line.lower():
                # Positive response indicators
                positive_indicators = ['yes', 'i think', 'maybe', 'could be', 'oh!']
                if any(indicator in context_line.lower() for indicator in positive_indicators):
                    effectiveness += 0.2

                # Elaborate response (indicates deeper thinking)
                if len(context_line.split()) > 5:
                    effectiveness += 0.1

        # Technique-specific effectiveness factors
        if technique_type == 'modeling_thinking' and 'i\'m' in line.lower():
            effectiveness += 0.15
        elif technique_type == 'collaborative_construction' and 'together' in line.lower():
            effectiveness += 0.15

        return min(1.0, effectiveness)

    def _assess_wait_time_quality(self, evidence: str) -> float:
        """Assess quality of wait time behavior"""
        quality = 0.5  # Base quality

        # Positive wait time indicators
        if any(indicator in evidence.lower() for indicator in ['take your time', 'think', 'moment']):
            quality += 0.3

        # Child processing indicators
        if any(indicator in evidence.lower() for indicator in ['um', 'hmm', 'well']):
            quality += 0.2

        return min(1.0, quality)

    def _assess_support_level(self, line: str) -> float:
        """Assess level of support provided (0=independent, 1=high support)"""
        support_level = 0.5  # Default moderate support

        # High support indicators
        high_support = ['let me show you', 'i\'ll do this', 'here\'s how', 'watch me']
        if any(indicator in line.lower() for indicator in high_support):
            support_level = 0.9

        # Low support indicators (encouraging independence)
        low_support = ['you try', 'what do you think', 'you can do this', 'by yourself']
        if any(indicator in line.lower() for indicator in low_support):
            support_level = 0.2

        return support_level

    def _assess_independence_encouragement(self, line: str) -> float:
        """Assess level of independence encouragement"""
        independence_score = 0.5

        independence_indicators = ['on your own', 'by yourself', 'you can do', 'you\'re ready']
        for indicator in independence_indicators:
            if indicator in line.lower():
                independence_score += 0.2

        return min(1.0, independence_score)

    def _analyze_support_progression(self, support_levels: List[Dict]) -> Dict[str, Any]:
        """Analyze progression of support levels throughout interaction"""
        if len(support_levels) < 2:
            return {'progression_score': 0.5, 'trend': 'insufficient_data'}

        # Calculate trend in support levels
        levels = [item['support_level'] for item in support_levels]

        # Simple trend analysis
        decreasing_support = 0
        for i in range(1, len(levels)):
            if levels[i] < levels[i-1]:
                decreasing_support += 1

        progression_score = decreasing_support / (len(levels) - 1) if len(levels) > 1 else 0.5

        trend = 'fading' if progression_score > 0.6 else 'consistent' if progression_score > 0.3 else 'increasing'

        return {
            'progression_score': progression_score,
            'trend': trend,
            'support_levels': levels,
            'fading_evidence': decreasing_support > 0
        }

    def _generate_overall_assessment(self, zpd_analysis: Dict, scaffolding_analysis: Dict,
                                   wait_time_analysis: Dict, fading_analysis: Dict) -> Dict[str, Any]:
        """Generate overall assessment of scaffolding and ZPD implementation"""
        # Calculate composite scores
        zpd_score = sum(data['average_confidence'] for data in zpd_analysis.values()) / max(1, len(zpd_analysis))
        scaffolding_score = sum(data['average_effectiveness'] for data in scaffolding_analysis.values()) / max(1, len(scaffolding_analysis))
        wait_time_score = 0.8 if any(data['frequency'] > 0 for data in wait_time_analysis.values()) else 0.3
        fading_score = fading_analysis.get('overall_fading_score', 0.5)

        overall_score = (zpd_score + scaffolding_score + wait_time_score + fading_score) / 4

        return {
            'overall_scaffolding_zpd_score': overall_score,
            'zpd_implementation_score': zpd_score,
            'scaffolding_technique_score': scaffolding_score,
            'wait_time_implementation_score': wait_time_score,
            'fading_support_score': fading_score,
            'assessment_summary': self._generate_assessment_summary(overall_score)
        }

    def _generate_assessment_summary(self, score: float) -> str:
        """Generate human-readable assessment summary"""
        if score >= 0.8:
            return "Excellent implementation of scaffolding and ZPD principles"
        elif score >= 0.6:
            return "Good use of scaffolding techniques with some ZPD alignment"
        elif score >= 0.4:
            return "Moderate scaffolding present, room for ZPD improvement"
        else:
            return "Limited scaffolding evidence, significant development opportunities"

    def _compile_research_citations(self) -> List[Dict[str, str]]:
        """Compile all research citations used in analysis"""
        citations = []

        # Add citations from ZPD indicators
        for zpd_data in self.zpd_indicators.values():
            citations.append({
                'citation': zpd_data['research'],
                'relevance': 'Zone of Proximal Development theory'
            })

        # Add citations from scaffolding techniques
        for technique_data in self.scaffolding_techniques.values():
            citations.append({
                'citation': technique_data['research'],
                'relevance': 'Scaffolding technique validation'
            })

        # Add citations from wait time research
        for behavior_data in self.wait_time_behaviors.values():
            citations.append({
                'citation': behavior_data['research'],
                'relevance': 'Wait time effectiveness research'
            })

        # Remove duplicates
        unique_citations = []
        seen_citations = set()
        for citation in citations:
            if citation['citation'] not in seen_citations:
                unique_citations.append(citation)
                seen_citations.add(citation['citation'])

        return unique_citations

    def _generate_recommendations(self, zpd_analysis: Dict,
                                scaffolding_analysis: Dict) -> List[str]:
        """Generate specific recommendations for improvement"""
        recommendations = []

        # ZPD-based recommendations
        for zpd_type, zpd_data in zpd_analysis.items():
            if zpd_data['frequency'] == 0:
                if zpd_type == 'appropriate_challenge':
                    recommendations.append("Consider providing challenges that build on children's existing knowledge")
                elif zpd_type == 'guided_discovery':
                    recommendations.append("Try using more open-ended questions to guide children's discoveries")
                elif zpd_type == 'developmental_matching':
                    recommendations.append("Adjust language complexity to match children's developmental level")

        # Scaffolding-based recommendations
        for technique_type, technique_data in scaffolding_analysis.items():
            if technique_data['frequency'] == 0:
                if technique_type == 'modeling_thinking':
                    recommendations.append("Model your thinking process aloud to help children learn metacognition")
                elif technique_type == 'graduated_prompting':
                    recommendations.append("Use increasingly specific prompts when children need support")
                elif technique_type == 'fading_support':
                    recommendations.append("Gradually reduce support to build children's independence")

        # Limit to top 3 most important recommendations
        return recommendations[:3]

    async def _fallback_analysis(self, transcript: str) -> Dict[str, Any]:
        """Fallback analysis when main analysis fails"""
        logger.warning("Using fallback scaffolding ZPD analysis")

        question_count = transcript.count('?')
        word_count = len(transcript.split())

        return {
            'zpd_indicators': {
                'appropriate_challenge': {'frequency': 0, 'average_confidence': 0.5},
                'guided_discovery': {'frequency': question_count, 'average_confidence': 0.6},
                'developmental_matching': {'frequency': 0, 'average_confidence': 0.5}
            },
            'scaffolding_techniques': {
                'modeling_thinking': {'frequency': 0, 'average_effectiveness': 0.5},
                'graduated_prompting': {'frequency': 0, 'average_effectiveness': 0.5},
                'collaborative_construction': {'frequency': 0, 'average_effectiveness': 0.5},
                'fading_support': {'frequency': 0, 'average_effectiveness': 0.5}
            },
            'overall_assessment': {
                'overall_scaffolding_zpd_score': 0.5,
                'assessment_summary': 'Analysis limited - unable to fully assess scaffolding techniques'
            },
            'recommendations': [
                'Consider using more scaffolding techniques to support learning',
                'Implement ZPD principles for appropriate challenge levels',
                'Model thinking processes to support children\'s development'
            ]
        }

# Singleton instance for integration
_scaffolding_zpd_analyzer = None

def get_scaffolding_zpd_analyzer() -> ScaffoldingZPDAnalyzer:
    """Get singleton scaffolding ZPD analyzer instance"""
    global _scaffolding_zpd_analyzer
    if _scaffolding_zpd_analyzer is None:
        _scaffolding_zpd_analyzer = ScaffoldingZPDAnalyzer()
    return _scaffolding_zpd_analyzer