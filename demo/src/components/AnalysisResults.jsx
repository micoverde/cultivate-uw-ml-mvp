import React, { useState } from 'react';
import {
  BarChart3,
  CheckCircle,
  AlertTriangle,
  Brain,
  MessageCircle,
  Clock,
  Users,
  FileText,
  TrendingUp,
  Download
} from 'lucide-react';
import CLASSDashboard from './CLASSDashboard';
import ScaffoldingVisualization from './ScaffoldingVisualization';
import EnhancedRecommendations from './EnhancedRecommendations';
// import ProfessionalVisualization from './ProfessionalVisualization';
import ExportManager from './ExportManager';

const AnalysisResults = ({ results, onStartNew, onStartNewScenario }) => {
  const [showExportModal, setShowExportModal] = useState(false);

  const {
    transcript_summary,
    ml_predictions,
    class_scores,
    scaffolding_analysis,
    recommendations,
    enhanced_recommendations,
    processing_time,
    completed_at
  } = results;

  // Extract original transcript from results if available (for ScaffoldingVisualization)
  const originalTranscript = results.original_transcript || '';

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
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50/30">
      {/* Professional Header */}
      <div className="bg-white shadow-sm border-b border-gray-200/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <div className="w-10 h-10 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center mr-3">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Cultivate Learning ML Analysis</h1>
                <p className="text-gray-600 text-sm">Professional Demo Interface</p>
              </div>
            </div>
            <div className="flex space-x-3">
              <button
                onClick={() => setShowExportModal(true)}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-medium text-sm flex items-center"
              >
                <Download className="h-4 w-4 mr-2" />
                Export Results
              </button>
              {onStartNewScenario && (
                <button
                  onClick={onStartNewScenario}
                  className="px-4 py-2 bg-white border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors font-medium text-sm"
                >
                  ← Back to Scenarios
                </button>
              )}
              <button
                onClick={onStartNew}
                className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors font-medium text-sm"
              >
                Analyze Another
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      {/* Note: ProfessionalVisualization will be integrated when merged from 3.2 branch */}

    <div className="max-w-6xl mx-auto p-6 space-y-8">
      {/* Traditional Analysis Details */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <div className="flex items-center mb-6">
          <BarChart3 className="h-6 w-6 text-indigo-600 mr-3" />
          <h2 className="text-xl font-semibold text-gray-900">Traditional Analysis Details</h2>
          <div className="ml-auto text-sm text-gray-500">
            Processing: {processing_time.toFixed(1)}s • {new Date(completed_at).toLocaleTimeString()}
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

          {/* CLASS Framework Dashboard */}
          {class_scores && (
            <CLASSDashboard
              classScores={class_scores}
              scaffoldingResults={scaffolding_analysis}
            />
          )}

          {/* Scaffolding & ZPD Analysis */}
          {scaffolding_analysis && (
            <ScaffoldingVisualization
              scaffoldingResults={scaffolding_analysis}
              originalTranscript={originalTranscript}
            />
          )}
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

          {/* Enhanced Recommendations */}
          <EnhancedRecommendations
            enhancedRecommendations={enhanced_recommendations}
            legacyRecommendations={recommendations}
          />

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

      {/* Export Modal */}
      <ExportManager
        results={results}
        scenarioContext={results.scenarioContext}
        isOpen={showExportModal}
        onClose={() => setShowExportModal(false)}
      />
      </div>
    </div>
  );
};

export default AnalysisResults;