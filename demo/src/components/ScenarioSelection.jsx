import React, { useState } from 'react';
import { Brain, ArrowLeft, Play, FileText, Users, Clock, Loader2 } from 'lucide-react';
import ScenarioGrid from './ScenarioGrid';
import { getAgeGroupLabel, getInteractionTypeLabel } from '../data/scenarios';

/**
 * ScenarioSelection Component
 *
 * Main interface for professional scenario selection and preview.
 * Integrates scenario grid with preview functionality for stakeholder presentations.
 *
 * Feature 3: Professional Demo Interface
 * Story 3.1: Select realistic educator scenarios
 *
 * Author: Claude (Partner-Level Microsoft SDE)
 * Issue: #48 - Scenario Selection Interface
 */

const ScenarioSelection = ({ onScenarioAnalyze, onBackToHome, isAnalyzing, analysisProgress }) => {
  const [selectedScenario, setSelectedScenario] = useState(null);
  const [previewMode, setPreviewMode] = useState(false);

  // Handle scenario selection from grid
  const handleScenarioSelect = (scenario) => {
    setSelectedScenario(scenario);
    setPreviewMode(true);
  };

  // Handle scenario analysis
  const handleAnalyzeScenario = () => {
    if (selectedScenario && onScenarioAnalyze) {
      // Convert scenario to transcript format expected by analysis API
      const analysisData = {
        transcript: selectedScenario.transcript,
        metadata: {
          educator_name: selectedScenario.participantInfo?.educatorExperience || null,
          child_age: parseInt(selectedScenario.ageGroup.split('-')[0]) || null,
          interaction_type: selectedScenario.interactionType === 'lesson' ? 'lesson' :
                           selectedScenario.interactionType === 'play' ? 'playtime' :
                           selectedScenario.interactionType === 'reading' ? 'reading' : 'general',
          duration_minutes: Math.ceil(selectedScenario.duration / 60)
        },
        scenarioContext: selectedScenario // Pass full scenario for enhanced analysis
      };

      onScenarioAnalyze(analysisData);
    }
  };

  // Handle back to grid
  const handleBackToGrid = () => {
    setPreviewMode(false);
    setSelectedScenario(null);
  };

  if (previewMode && selectedScenario) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">

        {/* Header */}
        <header className="bg-white shadow-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-6">
              <div className="flex items-center">
                <Brain className="h-8 w-8 text-indigo-600 mr-3" />
                <h1 className="text-2xl font-bold text-gray-900">Scenario Preview</h1>
              </div>
              <div className="flex space-x-4">
                <button
                  onClick={handleBackToGrid}
                  className="flex items-center text-gray-500 hover:text-gray-900 px-4 py-2 rounded-lg transition-colors"
                >
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  Back to Scenarios
                </button>
                <button
                  onClick={onBackToHome}
                  className="text-gray-500 hover:text-gray-900 px-4 py-2 rounded-lg transition-colors"
                >
                  ← Home
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Preview Content */}
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

            {/* Main Content */}
            <div className="lg:col-span-2 space-y-6">

              {/* Scenario Overview */}
              <div className="bg-white rounded-lg shadow-lg p-6">
                <div className="mb-6">
                  <h2 className="text-2xl font-bold text-gray-900 mb-3">
                    {selectedScenario.title}
                  </h2>
                  <p className="text-lg text-gray-700 leading-relaxed">
                    {selectedScenario.description}
                  </p>
                </div>

                {/* Key Details */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="text-center p-3 bg-blue-50 rounded-lg">
                    <Users className="h-6 w-6 text-blue-600 mx-auto mb-1" />
                    <div className="text-sm font-medium text-blue-900">
                      {getAgeGroupLabel(selectedScenario.ageGroup)}
                    </div>
                  </div>

                  <div className="text-center p-3 bg-green-50 rounded-lg">
                    <FileText className="h-6 w-6 text-green-600 mx-auto mb-1" />
                    <div className="text-sm font-medium text-green-900">
                      {getInteractionTypeLabel(selectedScenario.interactionType)}
                    </div>
                  </div>

                  <div className="text-center p-3 bg-purple-50 rounded-lg">
                    <Clock className="h-6 w-6 text-purple-600 mx-auto mb-1" />
                    <div className="text-sm font-medium text-purple-900">
                      {Math.ceil(selectedScenario.duration / 60)} minutes
                    </div>
                  </div>

                  <div className="text-center p-3 bg-indigo-50 rounded-lg">
                    <div className="text-lg font-bold text-indigo-600 mb-1">
                      {selectedScenario.expectedQuality === 'exemplary' ? 'A+' :
                       selectedScenario.expectedQuality === 'proficient' ? 'B+' :
                       selectedScenario.expectedQuality === 'developing' ? 'B-' : 'C'}
                    </div>
                    <div className="text-sm font-medium text-indigo-900 capitalize">
                      {selectedScenario.expectedQuality}
                    </div>
                  </div>
                </div>

                {/* Background Context */}
                <div className="bg-gray-50 rounded-lg p-4 mb-6">
                  <h3 className="font-semibold text-gray-900 mb-2">Educational Context</h3>
                  <p className="text-gray-700 text-sm leading-relaxed">
                    {selectedScenario.backgroundContext}
                  </p>
                </div>

                {/* Transcript Preview */}
                <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
                  <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                    <FileText className="h-4 w-4 mr-2" />
                    Interaction Transcript
                  </h3>

                  <div className="bg-white rounded p-4 font-mono text-sm leading-relaxed max-h-64 overflow-y-auto">
                    {selectedScenario.transcript.split('\n\n').map((paragraph, index) => (
                      <p key={index} className="mb-3 last:mb-0">
                        {paragraph.split('\n').map((line, lineIndex) => (
                          <span key={lineIndex}>
                            {line.startsWith('Teacher:') ? (
                              <span className="text-blue-600 font-semibold">{line}</span>
                            ) : line.startsWith('Child') || line.startsWith('Maria:') || line.startsWith('Marcus:') || line.startsWith('Emma:') || line.startsWith('Jake:') ? (
                              <span className="text-green-600 font-semibold">{line}</span>
                            ) : (
                              <span className="text-gray-600">{line}</span>
                            )}
                            {lineIndex < paragraph.split('\n').length - 1 && <br />}
                          </span>
                        ))}
                      </p>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Sidebar */}
            <div className="space-y-6">

              {/* Analysis Action */}
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  {isAnalyzing ? 'Analyzing Scenario...' : 'Ready to Analyze'}
                </h3>
                <p className="text-gray-600 mb-6 text-sm">
                  {isAnalyzing
                    ? 'Running ML analysis with real models. This may take 15-30 seconds...'
                    : 'Run our ML analysis on this scenario to see detailed insights about educational quality, questioning techniques, scaffolding, and recommendations for improvement.'
                  }
                </p>

                {/* Progress Bar */}
                {isAnalyzing && analysisProgress && (
                  <div className="mb-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-indigo-800">
                        {analysisProgress.message}
                      </span>
                      <span className="text-sm text-indigo-600">
                        {analysisProgress.progress}%
                      </span>
                    </div>
                    <div className="w-full bg-indigo-200 rounded-full h-2">
                      <div
                        className="bg-indigo-600 h-2 rounded-full transition-all duration-300 ease-out"
                        style={{ width: `${analysisProgress.progress}%` }}
                      />
                    </div>
                  </div>
                )}

                <button
                  onClick={handleAnalyzeScenario}
                  disabled={isAnalyzing}
                  className={`w-full px-6 py-3 rounded-lg font-semibold transition-colors flex items-center justify-center ${
                    isAnalyzing
                      ? 'bg-gray-400 text-gray-300 cursor-not-allowed'
                      : 'bg-indigo-600 text-white hover:bg-indigo-700'
                  }`}
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Play className="h-5 w-5 mr-2" />
                      Analyze This Scenario
                    </>
                  )}
                </button>
              </div>

              {/* Expected Insights */}
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Expected Insights</h3>
                <div className="space-y-3">
                  {selectedScenario.expectedInsights.map((insight, index) => (
                    <div key={index} className="flex items-start text-sm">
                      <span className="text-indigo-400 mr-2 mt-1">•</span>
                      <span className="text-gray-700">{insight}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Participant Information */}
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Participants</h3>

                <div className="space-y-4 text-sm">
                  <div>
                    <h4 className="font-medium text-gray-900 mb-1">Educator</h4>
                    <p className="text-gray-600">{selectedScenario.participantInfo.educatorExperience}</p>
                  </div>

                  <div>
                    <h4 className="font-medium text-gray-900 mb-1">Children</h4>
                    <ul className="text-gray-600 space-y-1">
                      {selectedScenario.participantInfo.childCharacteristics.map((characteristic, index) => (
                        <li key={index} className="flex items-start">
                          <span className="text-indigo-400 mr-1">•</span>
                          {characteristic}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div>
                    <h4 className="font-medium text-gray-900 mb-1">Session Goals</h4>
                    <ul className="text-gray-600 space-y-1">
                      {selectedScenario.participantInfo.sessionGoals.map((goal, index) => (
                        <li key={index} className="flex items-start">
                          <span className="text-indigo-400 mr-1">•</span>
                          {goal}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Main scenario selection grid view
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">

      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-indigo-600 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900">Professional Demo</h1>
            </div>
            <button
              onClick={onBackToHome}
              className="text-gray-500 hover:text-gray-900 px-4 py-2 rounded-lg transition-colors"
            >
              ← Home
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <ScenarioGrid onScenarioSelect={handleScenarioSelect} />
      </div>
    </div>
  );
};

export default ScenarioSelection;