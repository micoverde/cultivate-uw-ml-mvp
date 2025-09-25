import React, { useState, useEffect, useCallback } from 'react';
import { Send, AlertCircle, CheckCircle, Loader2, FileText, Clock, Users, Zap } from 'lucide-react';
import { API_ENDPOINTS } from '../config/api';

const TranscriptSubmission = ({ onAnalysisComplete }) => {
  const [transcript, setTranscript] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [validationErrors, setValidationErrors] = useState([]);
  const [validationWarnings, setValidationWarnings] = useState([]);
  const [analysisId, setAnalysisId] = useState(null);
  const [analysisProgress, setAnalysisProgress] = useState(null);
  const [isEnhancedMode, setIsEnhancedMode] = useState(false);
  const [websocket, setWebsocket] = useState(null);
  const [realTimeUpdates, setRealTimeUpdates] = useState([]);

  // Check if enhanced demo mode is enabled (Milestone #106)
  useEffect(() => {
    const demoMode = import.meta.env.VITE_DEMO_MODE;
    const enableWS = import.meta.env.VITE_ENABLE_WEBSOCKETS === 'true';
    setIsEnhancedMode(demoMode === 'enhanced' && enableWS);
  }, []);

  // Real-time validation
  const validateTranscriptRealTime = useCallback((text) => {
    const errors = [];
    const warnings = [];

    // Length validation
    if (text.length > 0 && text.length < 100) {
      errors.push(`Minimum 100 characters required (${text.length}/100)`);
    }
    if (text.length > 5000) {
      errors.push(`Maximum 5000 characters exceeded (${text.length}/5000)`);
    }

    // Speaker format validation
    const speakerPattern = /(Teacher|Educator|Child|Student|Adult):\s*\w+/i;
    if (text.length > 50 && !speakerPattern.test(text)) {
      errors.push('Include speaker labels (e.g., "Teacher: Hello!" or "Child: Hi!")');
    }

    // Turn count validation
    const turns = text.split('\n').filter(line => line.includes(':') && line.trim().length > 5);
    if (text.length > 100 && turns.length < 2) {
      errors.push('Include at least 2 conversational turns between speakers');
    }
    if (turns.length > 0 && turns.length < 4) {
      warnings.push('Recommend at least 4 conversational turns for better analysis');
    }

    // Educational quality indicators
    const questionCount = (text.match(/\?/g) || []).length;
    const wordCount = text.split(/\s+/).length;
    if (wordCount > 50 && questionCount === 0) {
      warnings.push('Consider including questions for richer educational analysis');
    }

    setValidationErrors(errors);
    setValidationWarnings(warnings);
  }, []);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      validateTranscriptRealTime(transcript);
    }, 300); // Debounce validation

    return () => clearTimeout(timeoutId);
  }, [transcript, validateTranscriptRealTime]);

  // Submit transcript for analysis
  const handleSubmit = async () => {
    if (validationErrors.length > 0) return;

    setIsSubmitting(true);
    try {
      const response = await fetch(API_ENDPOINTS.ANALYZE_TRANSCRIPT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ transcript }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail?.message || data.detail || 'Submission failed');
      }

      setAnalysisId(data.analysis_id);

      // Start polling for results
      pollAnalysisStatus(data.analysis_id);

    } catch (error) {
      console.error('Submission error:', error);
      setValidationErrors([error.message]);
      setIsSubmitting(false);
    }
  };

  // Poll analysis status
  const pollAnalysisStatus = async (id) => {
    try {
      const response = await fetch(API_ENDPOINTS.ANALYZE_STATUS(id));
      const status = await response.json();

      setAnalysisProgress(status);

      if (status.status === 'complete') {
        // Get final results
        const resultsResponse = await fetch(API_ENDPOINTS.ANALYZE_RESULTS(id));
        const results = await resultsResponse.json();

        setIsSubmitting(false);
        onAnalysisComplete?.(results);
      } else if (status.status === 'error') {
        setValidationErrors([status.message]);
        setIsSubmitting(false);
        setAnalysisProgress(null);
      } else {
        // Continue polling
        setTimeout(() => pollAnalysisStatus(id), 1000);
      }
    } catch (error) {
      console.error('Polling error:', error);
      setValidationErrors(['Analysis status check failed']);
      setIsSubmitting(false);
      setAnalysisProgress(null);
    }
  };

  const handleClear = () => {
    setTranscript('');
    setValidationErrors([]);
    setValidationWarnings([]);
    setAnalysisId(null);
    setAnalysisProgress(null);
    setIsSubmitting(false);
  };

  const loadSampleTranscript = () => {
    const sample = `Teacher: Good morning! What would you like to explore in our science center today?

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

    setTranscript(sample);
  };

  // Character count display
  const characterCount = transcript.length;
  const characterCountColor = characterCount < 100 ? 'text-red-500' :
                            characterCount > 4500 ? 'text-yellow-500' :
                            'text-green-500';

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
            <FileText className="h-6 w-6 text-indigo-600 mr-2" />
            <h2 className="text-2xl font-bold text-gray-900">Submit Educator Transcript</h2>
          </div>
          <button
            onClick={loadSampleTranscript}
            className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
          >
            Load Sample
          </button>
        </div>
        <p className="text-gray-600">
          Paste or type an educator-child interaction transcript for AI-powered analysis
        </p>
      </div>

      {/* Transcript Input */}
      <div className="mb-6">
        <div className="relative">
          <textarea
            value={transcript}
            onChange={(e) => setTranscript(e.target.value)}
            placeholder="Paste your educator-child interaction transcript here...

Example format:
Teacher: What do you think will happen if we mix these colors?
Child: Maybe they'll make a new color!
Teacher: Great thinking! Let's try it and see what happens..."
            className={`w-full h-64 p-4 border rounded-lg resize-y focus:ring-2 focus:ring-indigo-500 focus:border-transparent ${
              validationErrors.length > 0 ? 'border-red-300' :
              validationWarnings.length > 0 ? 'border-yellow-300' :
              'border-gray-300'
            }`}
            disabled={isSubmitting}
          />

          {/* Character count */}
          <div className="absolute bottom-2 right-2 text-sm">
            <span className={characterCountColor}>
              {characterCount}/5000
            </span>
          </div>
        </div>

        {/* Real-time statistics */}
        {transcript.length > 0 && (
          <div className="mt-2 flex items-center space-x-4 text-sm text-gray-500">
            <div className="flex items-center">
              <Users className="h-4 w-4 mr-1" />
              Words: {transcript.split(/\s+/).filter(w => w.length > 0).length}
            </div>
            <div className="flex items-center">
              <Clock className="h-4 w-4 mr-1" />
              Est. {Math.max(1, Math.round(transcript.split(/\s+/).length / 150))} min
            </div>
            <div>
              Turns: {transcript.split('\n').filter(line => line.includes(':') && line.trim().length > 5).length}
            </div>
          </div>
        )}
      </div>

      {/* Validation Messages */}
      {validationErrors.length > 0 && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-start">
            <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 mr-2 flex-shrink-0" />
            <div>
              <h3 className="text-sm font-medium text-red-800">Please fix the following issues:</h3>
              <ul className="mt-2 text-sm text-red-700 space-y-1">
                {validationErrors.map((error, index) => (
                  <li key={index}>• {error}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {validationWarnings.length > 0 && (
        <div className="mb-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-start">
            <AlertCircle className="h-5 w-5 text-yellow-500 mt-0.5 mr-2 flex-shrink-0" />
            <div>
              <h3 className="text-sm font-medium text-yellow-800">Suggestions for better analysis:</h3>
              <ul className="mt-2 text-sm text-yellow-700 space-y-1">
                {validationWarnings.map((warning, index) => (
                  <li key={index}>• {warning}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Enhanced Demo Mode Indicator */}
      {isEnhancedMode && (
        <div className="mb-4 p-3 bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-200 rounded-lg">
          <div className="flex items-center">
            <Zap className="h-5 w-5 text-purple-600 mr-2" />
            <span className="text-sm font-medium text-purple-800">
              Enhanced Demo Mode: Real-time Analysis Ready
            </span>
          </div>
          {realTimeUpdates.length > 0 && (
            <div className="mt-2 max-h-20 overflow-y-auto">
              {realTimeUpdates.slice(-3).map((update, index) => (
                <div key={index} className="text-xs text-purple-700 mb-1">
                  {update.message}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Analysis Progress */}
      {analysisProgress && (
        <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center">
              <Loader2 className="h-5 w-5 text-blue-500 animate-spin mr-2" />
              <span className="text-sm font-medium text-blue-800">
                {analysisProgress.message}
              </span>
            </div>
            <span className="text-sm text-blue-600">
              {analysisProgress.progress}%
            </span>
          </div>

          {/* Progress bar */}
          <div className="w-full bg-blue-200 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${analysisProgress.progress}%` }}
            />
          </div>

          {analysisProgress.estimated_seconds && (
            <div className="mt-2 text-xs text-blue-600">
              Estimated time remaining: ~{Math.max(0, Math.round(
                analysisProgress.estimated_seconds * (100 - analysisProgress.progress) / 100
              ))} seconds
            </div>
          )}
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex justify-between">
        <button
          onClick={handleClear}
          className="px-6 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
          disabled={isSubmitting}
        >
          Clear
        </button>

        <button
          onClick={handleSubmit}
          disabled={transcript.length === 0 || validationErrors.length > 0 || isSubmitting}
          className={`px-8 py-2 rounded-lg font-semibold flex items-center transition-colors ${
            transcript.length === 0 || validationErrors.length > 0 || isSubmitting
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-indigo-600 text-white hover:bg-indigo-700'
          }`}
        >
          {isSubmitting ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Send className="h-4 w-4 mr-2" />
              Analyze Transcript
            </>
          )}
        </button>
      </div>

      {/* Help Text */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="text-sm font-medium text-gray-900 mb-2">Tips for best results:</h3>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• Include clear speaker labels (Teacher:, Child:, Student:, etc.)</li>
          <li>• Capture natural back-and-forth conversation</li>
          <li>• Include questions, responses, and follow-up interactions</li>
          <li>• Aim for 4-10 conversational turns for comprehensive analysis</li>
          <li>• Remove any personal identifying information</li>
        </ul>
      </div>
    </div>
  );
};

export default TranscriptSubmission;