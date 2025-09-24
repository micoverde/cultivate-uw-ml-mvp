import React from 'react';
import {
  BarChart3,
  CheckCircle,
  AlertTriangle,
  Brain,
  MessageCircle,
  Clock,
  Users,
  FileText,
  TrendingUp
} from 'lucide-react';

const AnalysisResults = ({ results, onStartNew }) => {
  const {
    transcript_summary,
    ml_predictions,
    class_scores,
    recommendations,
    processing_time,
    completed_at
  } = results;

  // Format score display
  const formatScore = (score, isPercentage = true) => {
    if (isPercentage) {
      return `${Math.round(score * 100)}%`;
    }
    return score.toFixed(1);
  };

  // Get color based on score
  const getScoreColor = (score, threshold = 0.7) => {
    if (score >= threshold) return 'text-green-600';
    if (score >= threshold - 0.2) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBackground = (score, threshold = 0.7) => {
    if (score >= threshold) return 'bg-green-100';
    if (score >= threshold - 0.2) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Brain className="h-8 w-8 text-indigo-600 mr-3" />
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Analysis Results</h1>
              <p className="text-gray-600 mt-1">
                Completed in {processing_time.toFixed(1)}s • {new Date(completed_at).toLocaleTimeString()}
              </p>
            </div>
          </div>
          <button
            onClick={onStartNew}
            className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
          >
            Analyze Another
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Results Column */}
        <div className="lg:col-span-2 space-y-6">

          {/* ML Predictions */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center mb-6">
              <BarChart3 className="h-6 w-6 text-indigo-600 mr-2" />
              <h2 className="text-xl font-semibold text-gray-900">ML Analysis</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className={`p-4 rounded-lg ${getScoreBackground(ml_predictions.question_quality)}`}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-900">Question Quality</span>
                  <span className={`text-lg font-bold ${getScoreColor(ml_predictions.question_quality)}`}>
                    {formatScore(ml_predictions.question_quality)}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      ml_predictions.question_quality >= 0.7 ? 'bg-green-500' :
                      ml_predictions.question_quality >= 0.5 ? 'bg-yellow-500' :
                      'bg-red-500'
                    }`}
                    style={{ width: `${ml_predictions.question_quality * 100}%` }}
                  />
                </div>
              </div>

              <div className={`p-4 rounded-lg ${getScoreBackground(ml_predictions.wait_time_appropriate)}`}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-900">Wait Time</span>
                  <span className={`text-lg font-bold ${getScoreColor(ml_predictions.wait_time_appropriate)}`}>
                    {formatScore(ml_predictions.wait_time_appropriate)}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      ml_predictions.wait_time_appropriate >= 0.7 ? 'bg-green-500' :
                      ml_predictions.wait_time_appropriate >= 0.5 ? 'bg-yellow-500' :
                      'bg-red-500'
                    }`}
                    style={{ width: `${ml_predictions.wait_time_appropriate * 100}%` }}
                  />
                </div>
              </div>

              <div className={`p-4 rounded-lg ${getScoreBackground(ml_predictions.scaffolding_present)}`}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-900">Scaffolding</span>
                  <span className={`text-lg font-bold ${getScoreColor(ml_predictions.scaffolding_present)}`}>
                    {formatScore(ml_predictions.scaffolding_present)}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      ml_predictions.scaffolding_present >= 0.7 ? 'bg-green-500' :
                      ml_predictions.scaffolding_present >= 0.5 ? 'bg-yellow-500' :
                      'bg-red-500'
                    }`}
                    style={{ width: `${ml_predictions.scaffolding_present * 100}%` }}
                  />
                </div>
              </div>

              <div className={`p-4 rounded-lg ${getScoreBackground(ml_predictions.open_ended_questions)}`}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-900">Open-Ended Questions</span>
                  <span className={`text-lg font-bold ${getScoreColor(ml_predictions.open_ended_questions)}`}>
                    {formatScore(ml_predictions.open_ended_questions)}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      ml_predictions.open_ended_questions >= 0.7 ? 'bg-green-500' :
                      ml_predictions.open_ended_questions >= 0.5 ? 'bg-yellow-500' :
                      'bg-red-500'
                    }`}
                    style={{ width: `${ml_predictions.open_ended_questions * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* CLASS Framework Scores */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center mb-6">
              <TrendingUp className="h-6 w-6 text-indigo-600 mr-2" />
              <h2 className="text-xl font-semibold text-gray-900">CLASS Framework Scores</h2>
            </div>

            <div className="space-y-4">
              {[
                { key: 'emotional_support', label: 'Emotional Support', icon: '❤️' },
                { key: 'classroom_organization', label: 'Classroom Organization', icon: '📋' },
                { key: 'instructional_support', label: 'Instructional Support', icon: '🎓' },
                { key: 'overall_score', label: 'Overall Score', icon: '⭐', isBold: true }
              ].map(({ key, label, icon, isBold }) => {
                const score = class_scores[key];
                const isGood = score >= 4.0;

                return (
                  <div key={key} className={`flex items-center justify-between p-3 rounded-lg ${
                    isGood ? 'bg-green-50' : 'bg-yellow-50'
                  }`}>
                    <div className="flex items-center">
                      <span className="text-lg mr-2">{icon}</span>
                      <span className={`${isBold ? 'font-semibold' : 'font-medium'} text-gray-900`}>
                        {label}
                      </span>
                    </div>
                    <div className="flex items-center">
                      <span className={`text-2xl font-bold mr-2 ${
                        isGood ? 'text-green-600' : 'text-yellow-600'
                      }`}>
                        {formatScore(score, false)}
                      </span>
                      <span className="text-gray-500 text-sm">/ 7</span>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="mt-4 p-3 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">
                <strong>CLASS Framework:</strong> Classroom Assessment Scoring System measuring
                emotional support, classroom organization, and instructional support.
                Scores range from 1 (low) to 7 (high).
              </p>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">

          {/* Transcript Summary */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center mb-4">
              <FileText className="h-5 w-5 text-indigo-600 mr-2" />
              <h3 className="text-lg font-semibold text-gray-900">Summary</h3>
            </div>

            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Conversational Turns</span>
                <span className="font-medium">{transcript_summary.conversational_turns}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Word Count</span>
                <span className="font-medium">{transcript_summary.word_count}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Est. Duration</span>
                <span className="font-medium">{transcript_summary.estimated_duration_minutes.toFixed(1)} min</span>
              </div>
            </div>
          </div>

          {/* Recommendations */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center mb-4">
              <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
              <h3 className="text-lg font-semibold text-gray-900">Recommendations</h3>
            </div>

            <div className="space-y-3">
              {recommendations.map((recommendation, index) => (
                <div key={index} className="flex items-start p-3 bg-blue-50 rounded-lg">
                  <div className="flex-shrink-0 w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center mr-3 mt-0.5">
                    <span className="text-xs font-medium text-blue-600">{index + 1}</span>
                  </div>
                  <p className="text-sm text-gray-700 leading-relaxed">{recommendation}</p>
                </div>
              ))}
            </div>

            {recommendations.length === 0 && (
              <p className="text-sm text-gray-500 italic">
                Great work! No specific recommendations at this time.
              </p>
            )}
          </div>

          {/* Research Context */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center mb-4">
              <Brain className="h-5 w-5 text-purple-600 mr-2" />
              <h3 className="text-lg font-semibold text-gray-900">Research Basis</h3>
            </div>

            <div className="text-sm text-gray-600 space-y-2">
              <p>
                This analysis is based on established educational research from the
                <strong> CLASS framework</strong> and studies on effective educator-child interactions.
              </p>
              <p>
                Key factors include open-ended questioning, appropriate wait times,
                scaffolding techniques, and emotional support quality.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisResults;