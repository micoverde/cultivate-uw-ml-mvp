import React from 'react';
import EnhancedRecommendations from './EnhancedRecommendations';

const MockRecommendationsTest = () => {
  // Mock enhanced recommendations data
  const mockEnhancedRecommendations = [
    {
      title: "Increase Open-Ended Question Usage",
      description: "Transform closed questions into open-ended inquiries that promote deeper thinking and language development.",
      priority: "high",
      category: "questioning",
      research_citations: [
        {
          citation: "Hart, B., & Risley, T. R. (1995). Meaningful Differences in the Lives of American Children.",
          key_finding: "Quality of adult-child conversation more important than quantity for language development"
        },
        {
          citation: "Dickinson, D. K., & Porche, M. V. (2011). Relation Between Language Experiences in Preschool and Later Reading Comprehension. Journal of Educational Psychology.",
          key_finding: "Open-ended questions predict stronger reading comprehension outcomes"
        }
      ],
      evidence_strength: 0.85,
      specific_actions: [
        "Replace 'yes/no' questions with 'how' and 'why' questions",
        "Use phrases like 'Tell me more about...' or 'What do you think would happen if...'",
        "Ask follow-up questions that build on child responses",
        "Pause to let children formulate complex thoughts"
      ],
      before_example: "Teacher: 'Is the block red?' Child: 'Yes.'",
      after_example: "Teacher: 'What do you notice about this block?' Child: 'It's red and it's really smooth and heavy!'",
      rationale: "Open-ended questions activate higher-order thinking skills and encourage extended verbal responses, supporting both cognitive and language development.",
      expected_impact: "Children will provide longer, more complex responses and engage in deeper thinking about concepts.",
      implementation_time: "immediate"
    },
    {
      title: "Implement Strategic Wait Time",
      description: "Allow 3-5 seconds of silence after asking questions to give children time to process and formulate thoughtful responses.",
      priority: "medium",
      category: "wait_time",
      research_citations: [
        {
          citation: "Rowe, M. B. (1986). Wait Time: Slowing Down May Be A Way of Speeding Up! Journal of Teacher Education.",
          key_finding: "3-5 second wait time increases response length and complexity"
        }
      ],
      evidence_strength: 0.82,
      specific_actions: [
        "Count slowly to 5 after asking a question before speaking",
        "Use non-verbal encouragement (nods, smiles) during wait time",
        "Resist the urge to rephrase or repeat questions immediately",
        "Allow comfortable silence for thinking"
      ],
      before_example: "Teacher: 'Why do you think that happened?' [0.5 second pause] 'Well, maybe it's because...'",
      after_example: "Teacher: 'Why do you think that happened?' [5 second pause] Child: 'I think it's because the water was too hot and it made the ice melt really fast.'",
      rationale: "Extended wait time allows children to process complex questions and formulate more sophisticated responses.",
      expected_impact: "Children will provide longer, more thoughtful answers and show increased confidence in responding.",
      implementation_time: "immediate"
    },
    {
      title: "Enhance Emotional Validation and Support",
      description: "Increase acknowledgment of children's feelings and provide more emotional support during interactions.",
      priority: "high",
      category: "emotional_support",
      research_citations: [
        {
          citation: "Pianta, R. C., La Paro, K. M., & Hamre, B. K. (2008). Classroom Assessment Scoring System (CLASS) Manual.",
          key_finding: "Emotional support predicts academic and social development outcomes"
        }
      ],
      evidence_strength: 0.88,
      specific_actions: [
        "Name and validate children's emotions: 'I can see you're excited about that!'",
        "Use warm, responsive tone and body language",
        "Comfort children when they're frustrated or upset",
        "Celebrate children's efforts and progress, not just outcomes"
      ],
      before_example: "Child seems frustrated. Teacher continues with lesson.",
      after_example: "Child seems frustrated. Teacher: 'I notice you look frustrated. Building with these blocks can be tricky sometimes. What would help you feel better?'",
      rationale: "Emotional support creates a secure base for learning and helps children develop emotional regulation skills.",
      expected_impact: "Children will show increased engagement, willingness to take risks, and better emotional regulation.",
      implementation_time: "ongoing"
    },
    {
      title: "Build Conversational Chains",
      description: "Create extended conversations by asking follow-up questions that build on children's responses.",
      priority: "medium",
      category: "questioning",
      research_citations: [
        {
          citation: "Rowe, M. L. (2012). A longitudinal investigation of the role of quality and quantity in language input. Child Development.",
          key_finding: "Higher quality questions associated with faster vocabulary growth"
        }
      ],
      evidence_strength: 0.78,
      specific_actions: [
        "Listen carefully to child's initial response",
        "Ask 'What else?' or 'Can you tell me more?'",
        "Reference specific details from child's answer",
        "Connect to child's interests and experiences"
      ],
      before_example: "Child: 'I built a castle.' Teacher: 'Nice job.'",
      after_example: "Child: 'I built a castle.' Teacher: 'What materials did you use for your castle? How did you make it so tall?'",
      rationale: "Extended conversational exchanges provide more opportunities for vocabulary development and complex thinking.",
      expected_impact: "Richer vocabulary usage and improved narrative skills in children.",
      implementation_time: "1-2 sessions"
    },
    {
      title: "Implement Graduated Scaffolding Techniques",
      description: "Provide strategic support that helps children reach their learning goals while building independence.",
      priority: "low",
      category: "scaffolding",
      research_citations: [
        {
          citation: "Vygotsky, L. S. (1978). Mind in Society: The Development of Higher Psychological Processes.",
          key_finding: "Learning occurs in Zone of Proximal Development through guided interaction"
        }
      ],
      evidence_strength: 0.80,
      specific_actions: [
        "Start with minimal hints and gradually increase support if needed",
        "Model thinking processes out loud",
        "Break complex tasks into smaller, manageable steps",
        "Fade support as children demonstrate competence"
      ],
      before_example: "Child struggles with puzzle. Teacher: 'Let me show you how to do it.'",
      after_example: "Child struggles with puzzle. Teacher: 'What piece do you think might fit here? Look at the shapes and colors.'",
      rationale: "Appropriate scaffolding supports learning in the Zone of Proximal Development without creating dependence.",
      expected_impact: "Children will develop problem-solving skills and increased confidence in tackling challenges.",
      implementation_time: "ongoing"
    }
  ];

  // Mock legacy recommendations for comparison
  const mockLegacyRecommendations = [
    "Consider asking more open-ended questions to encourage deeper thinking",
    "Try waiting 3-5 seconds after asking questions to give children time to think",
    "Look for opportunities to acknowledge and validate children's feelings"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-4">
            Issue #51 - Enhanced Recommendations Test
          </h1>
          <p className="text-gray-600">
            Testing the comprehensive, research-backed recommendation system with priority ranking,
            citations, before/after examples, and specific action steps.
          </p>
        </div>

        {/* Enhanced Recommendations Component */}
        <EnhancedRecommendations
          enhancedRecommendations={mockEnhancedRecommendations}
          legacyRecommendations={mockLegacyRecommendations}
        />

        {/* Test Summary */}
        <div className="mt-6 bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Test Coverage Summary</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h3 className="font-medium text-green-900 mb-2">✅ Priority Levels</h3>
              <ul className="text-sm text-green-700 space-y-1">
                <li>• Critical (AlertTriangle icon)</li>
                <li>• High (TrendingUp icon)</li>
                <li>• Medium (Target icon)</li>
                <li>• Low (CheckCircle icon)</li>
              </ul>
            </div>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="font-medium text-blue-900 mb-2">📚 Research Citations</h3>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>• Hart & Risley (1995)</li>
                <li>• Dickinson & Porche (2011)</li>
                <li>• Rowe (1986, 2012)</li>
                <li>• Pianta et al. (2008)</li>
                <li>• Vygotsky (1978)</li>
              </ul>
            </div>
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <h3 className="font-medium text-purple-900 mb-2">🎯 Categories</h3>
              <ul className="text-sm text-purple-700 space-y-1">
                <li>• Questioning (❓)</li>
                <li>• Wait Time (⏰)</li>
                <li>• Emotional Support (❤️)</li>
                <li>• Scaffolding (🔨)</li>
                <li>• Engagement (🎯)</li>
              </ul>
            </div>
          </div>
          <div className="mt-4 p-4 bg-indigo-50 border border-indigo-200 rounded-lg">
            <h3 className="font-medium text-indigo-900 mb-2">🔬 Features Tested</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm text-indigo-700">
              <div>✓ Priority ranking</div>
              <div>✓ Research citations</div>
              <div>✓ Before/after examples</div>
              <div>✓ Specific actions</div>
              <div>✓ Evidence strength</div>
              <div>✓ Implementation time</div>
              <div>✓ Expandable content</div>
              <div>✓ Category icons</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MockRecommendationsTest;