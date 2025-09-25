import React, { useState } from 'react';
import { Brain, ArrowLeft, Send, Clock, User, AlertCircle, CheckCircle, Lightbulb, Target, Sparkles, Camera, Mic, Heart, BarChart3, Zap, Star, Award, Shield } from 'lucide-react';
import { mayaPuzzleScenario } from '../data/mayaScenario';

/**
 * EducatorResponseInput Component
 *
 * Implements the core Maya Puzzle Frustration scenario from the official UW demo script.
 * Users type their response to Maya's frustration and receive AI coaching feedback
 * across 5 evidence-based categories.
 *
 * PIVOT: From scenario analysis to educator response coaching
 *
 * Feature: User Response Evaluation System
 * Demo: DEMO 1 - Scenario-Based Response Analysis
 *
 * Author: Claude (Partner-Level Microsoft SDE)
 * Issue: #108 - PIVOT to User Response Evaluation
 */

const EducatorResponseInput = ({ onBackToHome, onResponseAnalysis, isAnalyzing, analysisProgress }) => {
  const [userResponse, setUserResponse] = useState('');
  const [responseValid, setResponseValid] = useState(false);
  const [showHints, setShowHints] = useState(false);

  // Validate response meets minimum requirements
  const validateResponse = (response) => {
    const isValid = response.length >= mayaPuzzleScenario.responseRequirements.minimumCharacters;
    setResponseValid(isValid);
    return isValid;
  };

  // Handle response text change
  const handleResponseChange = (e) => {
    const response = e.target.value;
    setUserResponse(response);
    validateResponse(response);
  };

  // Handle response submission
  const handleSubmitResponse = () => {
    if (!responseValid) return;

    const analysisData = {
      scenario_id: mayaPuzzleScenario.id,
      scenario_context: mayaPuzzleScenario.context,
      audio_transcript: mayaPuzzleScenario.audioTranscript,
      educator_response: userResponse,
      analysis_categories: mayaPuzzleScenario.analysisCategories,
      evidence_metrics: mayaPuzzleScenario.evidenceBasedMetrics,
      exemplar_response: mayaPuzzleScenario.exemplarResponse
    };

    onResponseAnalysis(analysisData);
  };

  const characterCount = userResponse.length;
  const minChars = mayaPuzzleScenario.responseRequirements.minimumCharacters;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 relative overflow-hidden">

      {/* Background decorative elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-32 w-80 h-80 bg-gradient-to-br from-indigo-200/30 to-purple-200/30 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-32 -left-32 w-80 h-80 bg-gradient-to-tr from-blue-200/30 to-teal-200/30 rounded-full blur-3xl"></div>
      </div>

      {/* Header */}
      <header className="bg-white/80 backdrop-blur-xl border-b border-white/20 shadow-lg shadow-indigo-100/25 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center group">
              <div className="relative">
                <Brain className="h-9 w-9 text-indigo-600 mr-4 drop-shadow-sm" />
                <div className="absolute -inset-1 bg-indigo-400/20 rounded-lg blur group-hover:bg-indigo-400/30 transition-colors"></div>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-900 to-indigo-600 bg-clip-text text-transparent">
                  AI-Powered Early Education Coaching
                </h1>
                <p className="text-sm text-gray-600 font-medium">Evidence-based feedback for educational excellence</p>
              </div>
            </div>
            <button
              onClick={onBackToHome}
              className="group flex items-center text-gray-600 hover:text-indigo-700 px-5 py-2.5 rounded-xl bg-white/60 hover:bg-white/80 border border-gray-200/60 hover:border-indigo-200 transition-all duration-200 shadow-sm hover:shadow-md backdrop-blur-sm"
              disabled={isAnalyzing}
            >
              <ArrowLeft className="h-4 w-4 mr-2 transition-transform group-hover:-translate-x-0.5" />
              <span className="font-medium">Home</span>
            </button>
          </div>
        </div>
      </header>

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">

          {/* Main Content Area */}
          <div className="lg:col-span-2 space-y-8">

            {/* Introduction */}
            <div className="group bg-white/70 backdrop-blur-xl rounded-2xl shadow-xl shadow-indigo-100/20 border border-white/40 p-8 hover:shadow-2xl hover:shadow-indigo-100/30 transition-all duration-300">
              <div className="flex items-center mb-4">
                <div className="p-2 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl mr-4 shadow-lg">
                  <Sparkles className="h-6 w-6 text-white" />
                </div>
                <h2 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                  Welcome to the Cultivate Learning Demo
                </h2>
              </div>
              <p className="text-gray-700 leading-relaxed text-lg">
                This tool uses machine learning trained on thousands of hours of high-quality early education
                interactions to provide evidence-based coaching. You'll practice responding to common classroom
                scenarios and receive personalized feedback.
              </p>
              <div className="mt-6 flex items-center text-indigo-600 text-sm font-medium">
                <Target className="h-4 w-4 mr-2" />
                <span>Powered by evidence-based pedagogical research</span>
              </div>
            </div>

            {/* Scenario Presentation */}
            <div className="group bg-white/70 backdrop-blur-xl rounded-2xl shadow-xl shadow-indigo-100/20 border border-white/40 p-8 hover:shadow-2xl hover:shadow-indigo-100/30 transition-all duration-300">
              <div className="flex items-center mb-6">
                <div className="p-3 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl mr-4 shadow-lg">
                  <User className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                  Scenario: Maya's Puzzle Frustration
                </h3>
              </div>

              {/* Visual Context Description */}
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-100 rounded-xl p-5 mb-6 shadow-inner">
                <div className="flex items-start">
                  <div className="p-1.5 bg-blue-100 rounded-lg mr-3 mt-0.5">
                    <Camera className="h-4 w-4 text-blue-600" />
                  </div>
                  <p className="text-blue-800 italic leading-relaxed font-medium">
                    {mayaPuzzleScenario.visualDescription}
                  </p>
                </div>
              </div>

              {/* Scenario Context */}
              <div className="bg-gradient-to-r from-slate-50 to-gray-50 border border-gray-100 rounded-xl p-5 mb-6 shadow-inner">
                <h4 className="font-bold text-gray-900 mb-3 flex items-center">
                  <div className="p-1.5 bg-gray-100 rounded-lg mr-3">
                    <AlertCircle className="h-4 w-4 text-gray-600" />
                  </div>
                  Context:
                </h4>
                <p className="text-gray-700 leading-relaxed">{mayaPuzzleScenario.context}</p>
              </div>

              {/* Audio Transcript */}
              <div className="bg-gradient-to-r from-amber-50 to-yellow-50 border-2 border-amber-200 rounded-xl p-5 shadow-inner">
                <h4 className="font-bold text-amber-900 mb-3 flex items-center">
                  <div className="p-2 bg-amber-100 rounded-lg mr-3">
                    <Mic className="h-5 w-5 text-amber-600" />
                  </div>
                  What Maya Says:
                </h4>
                <div className="bg-white/80 backdrop-blur-sm rounded-lg p-4 border border-amber-200">
                  <div className="text-amber-900 font-mono text-base whitespace-pre-line font-semibold tracking-wide">
                    {mayaPuzzleScenario.audioTranscript}
                  </div>
                </div>
              </div>
            </div>

            {/* Response Input Section */}
            <div className="group bg-white/70 backdrop-blur-xl rounded-2xl shadow-xl shadow-indigo-100/20 border border-white/40 p-8 hover:shadow-2xl hover:shadow-indigo-100/30 transition-all duration-300">
              <div className="flex items-center mb-6">
                <div className="p-3 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl mr-4 shadow-lg">
                  <Heart className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                  What would you say to Maya?
                </h3>
              </div>

              {/* Considerations */}
              <div className="mb-6">
                <p className="text-gray-700 font-semibold mb-3 flex items-center">
                  <Lightbulb className="h-4 w-4 mr-2 text-amber-500" />
                  Consider these pedagogical principles:
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {mayaPuzzleScenario.responseRequirements.considerations.map((consideration, index) => (
                    <div key={index} className="flex items-start p-3 bg-slate-50 rounded-lg border border-slate-200">
                      <div className="p-1 bg-indigo-100 rounded-full mr-3 mt-0.5">
                        <CheckCircle className="h-3 w-3 text-indigo-600" />
                      </div>
                      <span className="text-sm text-gray-700 font-medium leading-snug">{consideration}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Text Input */}
              <div className="relative mb-4">
                <textarea
                  value={userResponse}
                  onChange={handleResponseChange}
                  placeholder="Type your thoughtful, empathetic response to Maya here... (minimum 100 characters)"
                  className={`w-full h-40 p-5 border-2 rounded-2xl resize-none transition-all duration-300 text-base leading-relaxed ${
                    userResponse.length > 0
                      ? responseValid
                        ? 'border-emerald-300 bg-emerald-50/50 shadow-emerald-100/50 shadow-lg'
                        : 'border-amber-300 bg-amber-50/50 shadow-amber-100/50 shadow-lg'
                      : 'border-gray-200 bg-white/80 hover:border-indigo-300'
                  } focus:ring-4 focus:ring-indigo-100 focus:border-indigo-400 placeholder-gray-400`}
                  disabled={isAnalyzing}
                />
                {userResponse.length > 0 && (
                  <div className={`absolute top-4 right-4 p-2 rounded-full ${
                    responseValid ? 'bg-emerald-100' : 'bg-amber-100'
                  }`}>
                    {responseValid ? (
                      <CheckCircle className="h-5 w-5 text-emerald-600" />
                    ) : (
                      <Clock className="h-5 w-5 text-amber-600" />
                    )}
                  </div>
                )}
              </div>

              {/* Character Counter and Submit */}
              <div className="flex items-center justify-between">
                <div className={`flex items-center text-sm font-medium ${
                  characterCount < minChars ? 'text-amber-700' : 'text-emerald-700'
                }`}>
                  <div className={`p-1.5 rounded-full mr-2 ${
                    characterCount < minChars ? 'bg-amber-100' : 'bg-emerald-100'
                  }`}>
                    <BarChart3 className="h-4 w-4" />
                  </div>
                  <span>
                    {characterCount}/{minChars} characters
                    {characterCount < minChars && ` (${minChars - characterCount} more needed)`}
                  </span>
                </div>

                {/* Submit Button */}
                <button
                  onClick={handleSubmitResponse}
                  disabled={!responseValid || isAnalyzing}
                  className={`group px-8 py-3 rounded-2xl font-bold transition-all duration-300 flex items-center text-base shadow-lg ${
                    responseValid && !isAnalyzing
                      ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white hover:from-indigo-700 hover:to-purple-700 shadow-indigo-200 hover:shadow-xl hover:shadow-indigo-300/40 hover:-translate-y-0.5'
                      : 'bg-gray-200 text-gray-500 cursor-not-allowed shadow-gray-100'
                  }`}
                >
                  <Send className="h-5 w-5 mr-3 transition-transform group-hover:translate-x-0.5" />
                  {isAnalyzing ? 'Analyzing...' : 'Get AI Coaching Feedback'}
                </button>
              </div>
            </div>

            {/* Analysis Progress */}
            {isAnalyzing && analysisProgress && (
              <div className="bg-white/70 backdrop-blur-xl rounded-2xl shadow-xl shadow-indigo-100/20 border border-white/40 p-8 animate-pulse">
                <div className="flex items-center mb-6">
                  <div className="p-3 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl mr-4 shadow-lg animate-spin">
                    <Brain className="h-6 w-6 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                    AI Analysis in Progress...
                  </h3>
                </div>

                <p className="text-gray-600 mb-6 text-lg">
                  Analyzing your response using evidence-based early childhood pedagogy models trained on thousands of expert interactions.
                </p>

                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-base font-semibold text-indigo-800 flex items-center">
                      <Zap className="h-4 w-4 mr-2 text-amber-500" />
                      {analysisProgress.message}
                    </span>
                    <span className="text-lg font-bold text-indigo-600 bg-indigo-50 px-3 py-1 rounded-full">
                      {analysisProgress.progress}%
                    </span>
                  </div>
                  <div className="w-full bg-gradient-to-r from-indigo-100 to-purple-100 rounded-full h-3 shadow-inner">
                    <div
                      className="bg-gradient-to-r from-indigo-600 to-purple-600 h-3 rounded-full transition-all duration-500 ease-out shadow-lg"
                      style={{ width: `${analysisProgress.progress}%` }}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-8">

            {/* Helpful Hints */}
            <div className="group bg-white/70 backdrop-blur-xl rounded-2xl shadow-xl shadow-emerald-100/20 border border-white/40 p-6 hover:shadow-2xl hover:shadow-emerald-100/30 transition-all duration-300">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <div className="p-2 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-lg mr-3 shadow-lg">
                    <Lightbulb className="h-5 w-5 text-white" />
                  </div>
                  <h3 className="text-lg font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                    Coaching Tips
                  </h3>
                </div>
                <button
                  onClick={() => setShowHints(!showHints)}
                  className="px-3 py-1.5 bg-emerald-50 hover:bg-emerald-100 text-emerald-700 hover:text-emerald-800 text-sm font-medium rounded-lg transition-colors border border-emerald-200"
                >
                  {showHints ? 'Hide' : 'Show'} Hints
                </button>
              </div>

              {showHints && (
                <div className="space-y-3 animate-in slide-in-from-top-2 duration-300">
                  <div className="flex items-start p-3 bg-emerald-50 rounded-xl border border-emerald-100">
                    <div className="p-1 bg-emerald-500 rounded-full mr-3 mt-1">
                      <CheckCircle className="h-3 w-3 text-white" />
                    </div>
                    <span className="text-emerald-800 font-medium text-sm">Acknowledge Maya's feelings first</span>
                  </div>
                  <div className="flex items-start p-3 bg-emerald-50 rounded-xl border border-emerald-100">
                    <div className="p-1 bg-emerald-500 rounded-full mr-3 mt-1">
                      <CheckCircle className="h-3 w-3 text-white" />
                    </div>
                    <span className="text-emerald-800 font-medium text-sm">Notice what she accomplished</span>
                  </div>
                  <div className="flex items-start p-3 bg-emerald-50 rounded-xl border border-emerald-100">
                    <div className="p-1 bg-emerald-500 rounded-full mr-3 mt-1">
                      <CheckCircle className="h-3 w-3 text-white" />
                    </div>
                    <span className="text-emerald-800 font-medium text-sm">Offer choices and maintain autonomy</span>
                  </div>
                  <div className="flex items-start p-3 bg-emerald-50 rounded-xl border border-emerald-100">
                    <div className="p-1 bg-emerald-500 rounded-full mr-3 mt-1">
                      <CheckCircle className="h-3 w-3 text-white" />
                    </div>
                    <span className="text-emerald-800 font-medium text-sm">Use calm, supportive tone</span>
                  </div>
                </div>
              )}
            </div>

            {/* Analysis Categories Preview */}
            <div className="group bg-white/70 backdrop-blur-xl rounded-2xl shadow-xl shadow-indigo-100/20 border border-white/40 p-6 hover:shadow-2xl hover:shadow-indigo-100/30 transition-all duration-300">
              <div className="flex items-center mb-4">
                <div className="p-2 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg mr-3 shadow-lg">
                  <Target className="h-5 w-5 text-white" />
                </div>
                <h3 className="text-lg font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                  Analysis Categories
                </h3>
              </div>
              <p className="text-gray-600 text-sm mb-4 leading-relaxed">
                Your response will be evaluated across these evidence-based dimensions:
              </p>
              <div className="space-y-3 text-sm">
                {mayaPuzzleScenario.analysisCategories.map((category, index) => (
                  <div key={category.id} className="flex items-start p-3 bg-indigo-50 rounded-xl border border-indigo-100">
                    <div className="p-1.5 bg-indigo-500 rounded-lg mr-3 mt-0.5">
                      <Star className="h-3 w-3 text-white" />
                    </div>
                    <div>
                      <span className="font-bold text-indigo-900">{category.name}</span>
                      <p className="text-indigo-700 text-xs mt-1">{category.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Evidence-Based Research */}
            <div className="group bg-white/70 backdrop-blur-xl rounded-2xl shadow-xl shadow-amber-100/20 border border-white/40 p-6 hover:shadow-2xl hover:shadow-amber-100/30 transition-all duration-300">
              <div className="flex items-center mb-4">
                <div className="p-2 bg-gradient-to-br from-amber-500 to-orange-600 rounded-lg mr-3 shadow-lg">
                  <Award className="h-5 w-5 text-white" />
                </div>
                <h3 className="text-lg font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                  Research Foundation
                </h3>
              </div>
              <p className="text-gray-600 text-sm mb-4 leading-relaxed">
                This scenario is grounded in validated early childhood research:
              </p>
              <div className="space-y-3 text-sm">
                {mayaPuzzleScenario.researchBase.map((research, index) => (
                  <div key={index} className="flex items-start p-3 bg-amber-50 rounded-xl border border-amber-100">
                    <div className="p-1 bg-amber-500 rounded-full mr-3 mt-1">
                      <Shield className="h-3 w-3 text-white" />
                    </div>
                    <span className="text-amber-800 font-medium leading-snug">{research}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EducatorResponseInput;