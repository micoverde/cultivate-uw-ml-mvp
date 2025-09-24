import React from 'react';
import CLASSDashboard from './CLASSDashboard';
import ScaffoldingVisualization from './ScaffoldingVisualization';

const MockAnalysisTest = () => {
  // Mock data that matches the expected API structure
  const mockClassScores = {
    emotional_support: 4.2,
    classroom_organization: 3.8,
    instructional_support: 4.5,
    overall_score: 4.17,
    detailed_analysis: {
      emotional_support_breakdown: {
        dimension_scores: {
          positive_climate: 4.5,
          teacher_sensitivity: 4.2,
          regard_for_perspectives: 3.8
        }
      },
      classroom_organization_breakdown: {
        dimension_scores: {
          behavior_management: 4.0,
          productivity: 3.6
        }
      },
      instructional_support_breakdown: {
        dimension_scores: {
          concept_development: 4.2,
          quality_feedback: 4.8,
          language_modeling: 4.5
        }
      }
    }
  };

  const mockScaffoldingResults = {
    overall_assessment: {
      overall_scaffolding_zpd_score: 0.78,
      assessment_summary: 'Strong implementation of scaffolding techniques with good ZPD alignment. Educator demonstrates effective use of graduated prompting and appropriate challenge level.',
      zpd_implementation_score: 0.82,
      scaffolding_technique_score: 0.75,
      wait_time_implementation_score: 0.70,
      fading_support_score: 0.65
    },
    zpd_indicators: {
      appropriate_challenge: {
        frequency: 3,
        average_confidence: 0.85,
        description: 'Tasks and questions that match child development level',
        indicators: [
          {
            evidence: "What do you think will happen if we mix these colors?",
            line_number: 3,
            confidence: 0.9,
            research_backing: 'Vygotsky ZPD theory - appropriate challenge level'
          },
          {
            evidence: "That's a great start! Can you tell me more about what you see?",
            line_number: 7,
            confidence: 0.8,
            research_backing: 'Scaffolding research - building on child contributions'
          }
        ],
        research_backing: 'Based on Vygotsky\'s Zone of Proximal Development theory'
      },
      guided_discovery: {
        frequency: 2,
        average_confidence: 0.75,
        description: 'Supporting child-led exploration and discovery',
        indicators: [
          {
            evidence: "What do you notice about the leaf patterns?",
            line_number: 11,
            confidence: 0.8,
            research_backing: 'Constructivist learning approach'
          }
        ],
        research_backing: 'Constructivist learning research by Piaget and others'
      }
    },
    scaffolding_techniques: {
      modeling_thinking: {
        frequency: 2,
        average_effectiveness: 0.80,
        description: 'Demonstrating thought processes aloud',
        techniques_found: [
          {
            evidence: "I'm thinking about what might happen when we add water...",
            line_number: 15,
            effectiveness_score: 0.85
          }
        ],
        research_citation: 'Think-aloud protocols (Ericsson & Simon, 1993)'
      },
      graduated_prompting: {
        frequency: 4,
        average_effectiveness: 0.72,
        description: 'Providing increasingly specific hints and supports',
        techniques_found: [
          {
            evidence: "What do you see? ... Look closer at the colors ... Do you notice how blue and yellow are touching?",
            line_number: 8,
            effectiveness_score: 0.75
          },
          {
            evidence: "Can you try again? ... What if you hold it this way? ... Perfect!",
            line_number: 12,
            effectiveness_score: 0.70
          }
        ],
        research_citation: 'Gradual release of responsibility model (Pearson & Gallagher, 1983)'
      }
    },
    recommendations: [
      'Consider implementing more wait time after questions to allow deeper thinking',
      'Try using more open-ended follow-up questions to extend child reasoning',
      'Look for opportunities to fade support more gradually as child gains confidence'
    ]
  };

  const mockTranscript = `Teacher: Good morning! What would you like to explore in our science center today?

Child: I want to look at the magnifying glass!

Teacher: Great choice! What do you think we might discover with it?

Child: Maybe we can see tiny things bigger?

Teacher: That's exactly right! What small object would you like to examine first?

Child: This leaf looks interesting.

Teacher: Wonderful! Hold the magnifying glass steady and look closely. What do you notice about the leaf now?

Child: Wow! I can see all these lines and tiny dots!

Teacher: Those lines are called veins, and they help the leaf get water. What else do you observe?

Child: The edges are kind of fuzzy, and there are little holes!

Teacher: Excellent observation! You're thinking like a real scientist. What questions do you have about what you're seeing?`;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-4">
            Issue #50 - CLASS Dashboard Test
          </h1>
          <p className="text-gray-600">
            Testing the comprehensive CLASS Framework Dashboard and Scaffolding Analysis components
          </p>
        </div>

        {/* CLASS Framework Dashboard */}
        <CLASSDashboard
          classScores={mockClassScores}
          scaffoldingResults={mockScaffoldingResults}
        />

        {/* Scaffolding Visualization */}
        <div className="mt-6">
          <ScaffoldingVisualization
            scaffoldingResults={mockScaffoldingResults}
            originalTranscript={mockTranscript}
          />
        </div>
      </div>
    </div>
  );
};

export default MockAnalysisTest;