import React, { useState } from 'react';
import {
  CheckCircle,
  AlertTriangle,
  Star,
  BookOpen,
  Clock,
  Target,
  ChevronDown,
  ChevronRight,
  Quote,
  Lightbulb,
  TrendingUp,
  Award
} from 'lucide-react';

/**
 * EnhancedRecommendations Component
 *
 * Displays research-backed, actionable recommendations with priority ranking,
 * citations, before/after examples, and specific action steps.
 *
 * Author: Claude-4 (Partner-Level Microsoft SDE)
 * Issue: #51 - Story 2.5: Get actionable improvement recommendations
 */

const EnhancedRecommendations = ({ enhancedRecommendations, legacyRecommendations }) => {
  const [expandedRecommendation, setExpandedRecommendation] = useState(null);
  const [activeTab, setActiveTab] = useState('enhanced');

  // Priority styling configuration
  const getPriorityConfig = (priority) => {
    switch (priority) {
      case 'critical':
        return {
          color: 'text-red-600',
          bg: 'bg-red-50',
          border: 'border-red-200',
          icon: AlertTriangle,
          label: 'Critical'
        };
      case 'high':
        return {
          color: 'text-orange-600',
          bg: 'bg-orange-50',
          border: 'border-orange-200',
          icon: TrendingUp,
          label: 'High Priority'
        };
      case 'medium':
        return {
          color: 'text-yellow-600',
          bg: 'bg-yellow-50',
          border: 'border-yellow-200',
          icon: Target,
          label: 'Medium Priority'
        };
      case 'low':
        return {
          color: 'text-green-600',
          bg: 'bg-green-50',
          border: 'border-green-200',
          icon: CheckCircle,
          label: 'Low Priority'
        };
      default:
        return {
          color: 'text-blue-600',
          bg: 'bg-blue-50',
          border: 'border-blue-200',
          icon: Lightbulb,
          label: 'Suggestion'
        };
    }
  };

  // Category styling configuration
  const getCategoryIcon = (category) => {
    switch (category) {
      case 'questioning':
        return '❓';
      case 'wait_time':
        return '⏰';
      case 'emotional_support':
        return '❤️';
      case 'scaffolding':
        return '🔨';
      case 'classroom_organization':
        return '📋';
      case 'language_modeling':
        return '💬';
      case 'engagement':
        return '🎯';
      default:
        return '💡';
    }
  };

  const getImplementationTimeIcon = (time) => {
    switch (time) {
      case 'immediate':
        return '⚡';
      case '1-2 sessions':
        return '📅';
      case 'ongoing':
        return '🔄';
      default:
        return '⏱️';
    }
  };

  const toggleExpanded = (index) => {
    setExpandedRecommendation(expandedRecommendation === index ? null : index);
  };

  // Enhanced recommendations display
  const renderEnhancedRecommendations = () => (
    <div className="space-y-6">
      {enhancedRecommendations.map((recommendation, index) => {
        const priorityConfig = getPriorityConfig(recommendation.priority);
        const PriorityIcon = priorityConfig.icon;
        const isExpanded = expandedRecommendation === index;

        return (
          <div
            key={index}
            className={`border ${priorityConfig.border} rounded-lg p-6 ${priorityConfig.bg} transition-all duration-200 hover:shadow-md`}
          >
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-start space-x-3 flex-1">
                <div className="flex-shrink-0">
                  <PriorityIcon className={`h-6 w-6 ${priorityConfig.color} mt-0.5`} />
                </div>
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-2">
                    <h3 className="text-lg font-semibold text-gray-900">
                      {recommendation.title}
                    </h3>
                    <span className="text-2xl">{getCategoryIcon(recommendation.category)}</span>
                  </div>
                  <div className="flex items-center space-x-4 mb-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${priorityConfig.color} bg-white`}>
                      {priorityConfig.label}
                    </span>
                    <div className="flex items-center text-sm text-gray-600">
                      <Star className="h-4 w-4 mr-1" />
                      Evidence: {Math.round(recommendation.evidence_strength * 100)}%
                    </div>
                    <div className="flex items-center text-sm text-gray-600">
                      <span className="mr-1">{getImplementationTimeIcon(recommendation.implementation_time)}</span>
                      {recommendation.implementation_time}
                    </div>
                  </div>
                  <p className="text-gray-700 leading-relaxed">{recommendation.description}</p>
                </div>
              </div>
              <button
                onClick={() => toggleExpanded(index)}
                className="flex-shrink-0 p-1 hover:bg-white hover:bg-opacity-50 rounded-full transition-colors ml-4"
              >
                {isExpanded ? (
                  <ChevronDown className="h-5 w-5 text-gray-600" />
                ) : (
                  <ChevronRight className="h-5 w-5 text-gray-600" />
                )}
              </button>
            </div>

            {/* Expanded content */}
            {isExpanded && (
              <div className="border-t pt-6 space-y-6 bg-white bg-opacity-50 rounded-lg p-4">
                {/* Specific Actions */}
                <div>
                  <h4 className="font-semibold text-gray-900 mb-3 flex items-center">
                    <CheckCircle className="h-5 w-5 mr-2 text-green-600" />
                    Specific Actions
                  </h4>
                  <ul className="space-y-2">
                    {recommendation.specific_actions.map((action, actionIndex) => (
                      <li key={actionIndex} className="flex items-start text-sm text-gray-700">
                        <span className="flex-shrink-0 w-6 h-6 bg-green-100 rounded-full flex items-center justify-center mr-3 mt-0.5">
                          <span className="text-xs font-medium text-green-600">{actionIndex + 1}</span>
                        </span>
                        {action}
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Before/After Examples */}
                {(recommendation.before_example || recommendation.after_example) && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {recommendation.before_example && (
                      <div className="border border-red-200 rounded-lg p-4 bg-red-50">
                        <h5 className="font-medium text-red-900 mb-2 flex items-center">
                          <Quote className="h-4 w-4 mr-2" />
                          Before (Less Effective)
                        </h5>
                        <p className="text-sm text-red-800 italic">"{recommendation.before_example}"</p>
                      </div>
                    )}
                    {recommendation.after_example && (
                      <div className="border border-green-200 rounded-lg p-4 bg-green-50">
                        <h5 className="font-medium text-green-900 mb-2 flex items-center">
                          <Award className="h-4 w-4 mr-2" />
                          After (More Effective)
                        </h5>
                        <p className="text-sm text-green-800 italic">"{recommendation.after_example}"</p>
                      </div>
                    )}
                  </div>
                )}

                {/* Rationale and Impact */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h5 className="font-medium text-gray-900 mb-2 flex items-center">
                      <Lightbulb className="h-4 w-4 mr-2 text-blue-600" />
                      Why This Matters
                    </h5>
                    <p className="text-sm text-gray-700">{recommendation.rationale}</p>
                  </div>
                  <div>
                    <h5 className="font-medium text-gray-900 mb-2 flex items-center">
                      <Target className="h-4 w-4 mr-2 text-purple-600" />
                      Expected Impact
                    </h5>
                    <p className="text-sm text-gray-700">{recommendation.expected_impact}</p>
                  </div>
                </div>

                {/* Research Citations */}
                {recommendation.research_citations && recommendation.research_citations.length > 0 && (
                  <div>
                    <h5 className="font-medium text-gray-900 mb-3 flex items-center">
                      <BookOpen className="h-4 w-4 mr-2 text-indigo-600" />
                      Research Support
                    </h5>
                    <div className="space-y-3">
                      {recommendation.research_citations.map((citation, citIndex) => (
                        <div key={citIndex} className="border-l-4 border-indigo-200 pl-4 bg-indigo-50 p-3 rounded-r-lg">
                          <p className="text-sm font-medium text-indigo-900">{citation.citation}</p>
                          {citation.key_finding && (
                            <p className="text-sm text-indigo-700 mt-1">
                              <strong>Key Finding:</strong> {citation.key_finding}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );

  // Legacy recommendations display (for comparison)
  const renderLegacyRecommendations = () => (
    <div className="space-y-3">
      {legacyRecommendations.map((recommendation, index) => (
        <div key={index} className="flex items-start p-3 bg-blue-50 rounded-lg border border-blue-200">
          <div className="flex-shrink-0 w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center mr-3 mt-0.5">
            <span className="text-xs font-medium text-blue-600">{index + 1}</span>
          </div>
          <p className="text-sm text-gray-700 leading-relaxed">{recommendation}</p>
        </div>
      ))}
    </div>
  );

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900 flex items-center">
          <Lightbulb className="h-6 w-6 text-indigo-600 mr-2" />
          Improvement Recommendations
        </h2>
        {enhancedRecommendations && enhancedRecommendations.length > 0 && (
          <div className="bg-indigo-100 text-indigo-800 px-3 py-1 rounded-full text-sm font-medium">
            Research-Backed Insights
          </div>
        )}
      </div>

      {/* Tab Navigation */}
      {enhancedRecommendations && enhancedRecommendations.length > 0 && (
        <div className="mb-6 border-b border-gray-200">
          <nav className="flex space-x-8">
            <button
              onClick={() => setActiveTab('enhanced')}
              className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'enhanced'
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Enhanced Recommendations
            </button>
            {legacyRecommendations && legacyRecommendations.length > 0 && (
              <button
                onClick={() => setActiveTab('legacy')}
                className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'legacy'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Quick Tips
              </button>
            )}
          </nav>
        </div>
      )}

      {/* Content */}
      {activeTab === 'enhanced' && enhancedRecommendations && enhancedRecommendations.length > 0 ? (
        renderEnhancedRecommendations()
      ) : activeTab === 'legacy' && legacyRecommendations && legacyRecommendations.length > 0 ? (
        renderLegacyRecommendations()
      ) : enhancedRecommendations && enhancedRecommendations.length > 0 ? (
        renderEnhancedRecommendations()
      ) : legacyRecommendations && legacyRecommendations.length > 0 ? (
        renderLegacyRecommendations()
      ) : (
        <div className="text-center py-8">
          <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Excellent Work!</h3>
          <p className="text-gray-600">
            Your interaction demonstrates high-quality educational practices.
            Continue using these effective strategies with your students.
          </p>
        </div>
      )}
    </div>
  );
};

export default EnhancedRecommendations;