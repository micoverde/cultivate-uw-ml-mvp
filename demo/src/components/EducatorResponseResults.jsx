import React from 'react';
import {
  BarChart3,
  CheckCircle,
  AlertTriangle,
  Brain,
  MessageCircle,
  Clock,
  TrendingUp,
  Award,
  Target,
  Lightbulb,
  Star,
  ArrowLeft,
  Zap
} from 'lucide-react';

/**
 * EducatorResponseResults Component
 *
 * Displays comprehensive coaching feedback for educator responses to scenarios.
 * Shows 5-category analysis, evidence-based metrics, and personalized recommendations.
 *
 * PIVOT: From transcript analysis to educator response coaching
 *
 * Feature: User Response Evaluation Results Display
 * Demo: DEMO 1 - Maya Scenario Coaching Results
 *
 * Author: Claude (Partner-Level Microsoft SDE)
 * Issue: #113 - Maya Scenario Response Coaching System
 */

const EducatorResponseResults = ({ results, onStartNew }) => {
  const {
    analysis_id,
    scenario_id,
    category_scores = [],
    overall_coaching_score,
    evidence_metrics = [],
    coaching_recommendations = [],
    strengths_identified = [],
    growth_opportunities = [],
    suggested_response,
    processing_time,
    confidence_score,
    completed_at
  } = results;

  // Get color based on score (0-10 scale)
  const getScoreColor = (score) => {
    if (score >= 8) return 'text-emerald-600';
    if (score >= 6) return 'text-amber-600';
    return 'text-red-600';
  };

  const getScoreBackground = (score) => {
    if (score >= 8) return 'bg-emerald-50 border-emerald-200';
    if (score >= 6) return 'bg-amber-50 border-amber-200';
    return 'bg-red-50 border-red-200';
  };

  const getProgressColor = (score) => {
    if (score >= 8) return 'from-emerald-500 to-emerald-600';
    if (score >= 6) return 'from-amber-500 to-amber-600';
    return 'from-red-500 to-red-600';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 relative overflow-hidden">

      {/* Background decorative elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-32 w-80 h-80 bg-gradient-to-br from-emerald-200/30 to-teal-200/30 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-32 -left-32 w-80 h-80 bg-gradient-to-tr from-indigo-200/30 to-purple-200/30 rounded-full blur-3xl"></div>
      </div>

      {/* Header */}
      <header className="bg-white/80 backdrop-blur-xl border-b border-white/20 shadow-lg shadow-emerald-100/25 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center group">
              <div className="relative">
                <Award className="h-9 w-9 text-emerald-600 mr-4 drop-shadow-sm" />
                <div className="absolute -inset-1 bg-emerald-400/20 rounded-lg blur group-hover:bg-emerald-400/30 transition-colors"></div>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-emerald-900 to-emerald-600 bg-clip-text text-transparent">
                  Your Coaching Feedback Results
                </h1>
                <p className="text-sm text-gray-600 font-medium">Maya's Puzzle Frustration Scenario Analysis</p>
              </div>
            </div>
            <button
              onClick={onStartNew}
              className="group flex items-center text-gray-600 hover:text-emerald-700 px-5 py-2.5 rounded-xl bg-white/60 hover:bg-white/80 border border-gray-200/60 hover:border-emerald-200 transition-all duration-200 shadow-sm hover:shadow-md backdrop-blur-sm"
            >
              <ArrowLeft className="h-4 w-4 mr-2 transition-transform group-hover:-translate-x-0.5" />
              <span className="font-medium">Try Another Scenario</span>
            </button>
          </div>
        </div>
      </header>

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">

          {/* Main Results Area */}
          <div className="lg:col-span-2 space-y-8">

            {/* Overall Score Card */}
            <div className="group bg-white/70 backdrop-blur-xl rounded-2xl shadow-xl shadow-emerald-100/20 border border-white/40 p-8 hover:shadow-2xl hover:shadow-emerald-100/30 transition-all duration-300">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center">
                  <div className="p-3 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl mr-4 shadow-lg">
                    <TrendingUp className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                      Overall Coaching Score
                    </h2>
                    <p className="text-gray-600">Comprehensive pedagogical effectiveness</p>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-4xl font-bold ${getScoreColor(overall_coaching_score)}`}>
                    {overall_coaching_score}/10
                  </div>
                  <div className="text-sm text-gray-500">
                    {confidence_score && `${Math.round(confidence_score * 100)}% confidence`}
                  </div>
                </div>
              </div>

              <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
                <div
                  className={`bg-gradient-to-r ${getProgressColor(overall_coaching_score)} h-3 rounded-full transition-all duration-1000 ease-out shadow-lg`}
                  style={{ width: `${(overall_coaching_score / 10) * 100}%` }}
                />
              </div>

              <div className="flex items-center text-gray-600 text-sm">
                <Clock className="h-4 w-4 mr-2" />
                <span>Analysis completed in {processing_time?.toFixed(2)}s</span>
              </div>
            </div>

            {/* Category Scores */}
            <div className="group bg-white/70 backdrop-blur-xl rounded-2xl shadow-xl shadow-indigo-100/20 border border-white/40 p-8 hover:shadow-2xl hover:shadow-indigo-100/30 transition-all duration-300">
              <div className="flex items-center mb-6">
                <div className="p-3 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl mr-4 shadow-lg">
                  <BarChart3 className="h-6 w-6 text-white" />
                </div>
                <h2 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                  Category Analysis
                </h2>
              </div>

              <div className="space-y-6">
                {category_scores.map((category, index) => (
                  <div key={category.category_id} className={`p-5 rounded-xl border-2 ${getScoreBackground(category.score)}`}>
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="font-bold text-gray-900">{category.category_name}</h3>
                      <span className={`text-xl font-bold ${getScoreColor(category.score)}`}>
                        {category.score}/10
                      </span>
                    </div>

                    <div className="w-full bg-white/60 rounded-full h-2 mb-4">
                      <div
                        className={`bg-gradient-to-r ${getProgressColor(category.score)} h-2 rounded-full transition-all duration-700 delay-${index * 100}`}
                        style={{ width: `${(category.score / 10) * 100}%` }}
                      />
                    </div>

                    <p className="text-gray-700 text-sm mb-3">{category.feedback}</p>

                    {category.strengths && category.strengths.length > 0 && (
                      <div className="mb-3">
                        <h4 className="font-semibold text-emerald-800 text-sm mb-2 flex items-center">
                          <CheckCircle className="h-4 w-4 mr-1" />
                          Strengths
                        </h4>
                        <ul className="space-y-1">
                          {category.strengths.map((strength, idx) => (
                            <li key={idx} className="text-emerald-700 text-sm flex items-start">
                              <span className="text-emerald-500 mr-2">•</span>
                              {strength}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {category.growth_areas && category.growth_areas.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-amber-800 text-sm mb-2 flex items-center">
                          <TrendingUp className="h-4 w-4 mr-1" />
                          Growth Areas
                        </h4>
                        <ul className="space-y-1">
                          {category.growth_areas.map((area, idx) => (
                            <li key={idx} className="text-amber-700 text-sm flex items-start">
                              <span className="text-amber-500 mr-2">•</span>
                              {area}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Evidence-Based Metrics */}
            {evidence_metrics.length > 0 && (
              <div className="group bg-white/70 backdrop-blur-xl rounded-2xl shadow-xl shadow-purple-100/20 border border-white/40 p-8 hover:shadow-2xl hover:shadow-purple-100/30 transition-all duration-300">
                <div className="flex items-center mb-6">
                  <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl mr-4 shadow-lg">
                    <Brain className="h-6 w-6 text-white" />
                  </div>
                  <h2 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                    Evidence-Based Analysis
                  </h2>
                </div>

                <div className="space-y-4">
                  {evidence_metrics.map((metric, index) => (
                    <div key={metric.strategy} className="p-4 bg-purple-50 rounded-xl border border-purple-200">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-bold text-purple-900">{metric.name}</h3>
                        <div className="flex items-center">
                          {metric.detected && (
                            <CheckCircle className="h-5 w-5 text-emerald-600 mr-2" />
                          )}
                          <span className="text-purple-700 font-semibold">
                            {metric.effectiveness}% effectiveness
                          </span>
                        </div>
                      </div>
                      <p className="text-purple-700 text-sm">{metric.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-8">

            {/* Strengths */}
            {strengths_identified.length > 0 && (
              <div className="group bg-white/70 backdrop-blur-xl rounded-2xl shadow-xl shadow-emerald-100/20 border border-white/40 p-6 hover:shadow-2xl hover:shadow-emerald-100/30 transition-all duration-300">
                <div className="flex items-center mb-4">
                  <div className="p-2 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-lg mr-3 shadow-lg">
                    <Star className="h-5 w-5 text-white" />
                  </div>
                  <h3 className="text-lg font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                    Key Strengths
                  </h3>
                </div>
                <div className="space-y-3">
                  {strengths_identified.map((strength, index) => (
                    <div key={index} className="flex items-start p-3 bg-emerald-50 rounded-xl border border-emerald-100">
                      <CheckCircle className="h-4 w-4 text-emerald-600 mr-3 mt-0.5 flex-shrink-0" />
                      <span className="text-emerald-800 font-medium text-sm">{strength}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Growth Opportunities */}
            {growth_opportunities.length > 0 && (
              <div className="group bg-white/70 backdrop-blur-xl rounded-2xl shadow-xl shadow-amber-100/20 border border-white/40 p-6 hover:shadow-2xl hover:shadow-amber-100/30 transition-all duration-300">
                <div className="flex items-center mb-4">
                  <div className="p-2 bg-gradient-to-br from-amber-500 to-orange-600 rounded-lg mr-3 shadow-lg">
                    <TrendingUp className="h-5 w-5 text-white" />
                  </div>
                  <h3 className="text-lg font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                    Growth Opportunities
                  </h3>
                </div>
                <div className="space-y-3">
                  {growth_opportunities.map((opportunity, index) => (
                    <div key={index} className="flex items-start p-3 bg-amber-50 rounded-xl border border-amber-100">
                      <Lightbulb className="h-4 w-4 text-amber-600 mr-3 mt-0.5 flex-shrink-0" />
                      <span className="text-amber-800 font-medium text-sm">{opportunity}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Coaching Recommendations */}
            {coaching_recommendations.length > 0 && (
              <div className="group bg-white/70 backdrop-blur-xl rounded-2xl shadow-xl shadow-indigo-100/20 border border-white/40 p-6 hover:shadow-2xl hover:shadow-indigo-100/30 transition-all duration-300">
                <div className="flex items-center mb-4">
                  <div className="p-2 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg mr-3 shadow-lg">
                    <Target className="h-5 w-5 text-white" />
                  </div>
                  <h3 className="text-lg font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                    Coaching Recommendations
                  </h3>
                </div>
                <div className="space-y-4">
                  {coaching_recommendations.map((rec, index) => (
                    <div key={index} className="p-4 bg-indigo-50 rounded-xl border border-indigo-100">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-bold text-indigo-900 text-sm">{rec.category}</h4>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          rec.priority === 'high' ? 'bg-red-100 text-red-800' :
                          rec.priority === 'medium' ? 'bg-amber-100 text-amber-800' :
                          'bg-green-100 text-green-800'
                        }`}>
                          {rec.priority} priority
                        </span>
                      </div>
                      <p className="text-indigo-800 font-medium text-sm mb-2">{rec.recommendation}</p>
                      <p className="text-indigo-600 text-xs">{rec.evidence_base}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Suggested Response */}
            {suggested_response && (
              <div className="group bg-white/70 backdrop-blur-xl rounded-2xl shadow-xl shadow-violet-100/20 border border-white/40 p-6 hover:shadow-2xl hover:shadow-violet-100/30 transition-all duration-300">
                <div className="flex items-center mb-4">
                  <div className="p-2 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg mr-3 shadow-lg">
                    <MessageCircle className="h-5 w-5 text-white" />
                  </div>
                  <h3 className="text-lg font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                    Enhanced Response
                  </h3>
                </div>
                <div className="bg-violet-50 border border-violet-200 rounded-xl p-4">
                  <p className="text-violet-900 font-medium leading-relaxed text-sm">
                    "{suggested_response}"
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default EducatorResponseResults;