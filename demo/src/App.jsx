import React, { useState, useEffect } from 'react';
import { Brain, Users, Camera, Mic, BarChart3, Heart, Sparkles, Sun, Moon, ChevronRight, Star, Award, Zap, Target, Shield } from 'lucide-react';
import TranscriptSubmission from './components/TranscriptSubmission';
import AnalysisResults from './components/AnalysisResults';
import EducatorResponseResults from './components/EducatorResponseResults';
import MockAnalysisTest from './components/MockAnalysisTest';
import MockRecommendationsTest from './components/MockRecommendationsTest';
import ScenarioSelection from './components/ScenarioSelection';
import EducatorResponseInput from './components/EducatorResponseInput';
import { API_ENDPOINTS } from './config/api';
import { monitoring } from './config/monitoring';
import SecurityManager from './config/security';

function App() {
  const [currentView, setCurrentView] = useState('home'); // 'home', 'demo', 'scenarios', 'educator-response', 'results', 'educator-response-results', 'recommendations'
  const [analysisResults, setAnalysisResults] = useState(null);
  const [educatorResponseResults, setEducatorResponseResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(null);
  const [isDarkMode, setIsDarkMode] = useState(() => {
    // Check localStorage or default to dark mode
    const saved = localStorage.getItem('darkMode');
    return saved !== null ? saved === 'true' : true;
  });

  useEffect(() => {
    // Initialize security controls
    try {
      SecurityManager.initialize();
      console.log('Security controls initialized');
    } catch (error) {
      console.error('Security initialization failed:', error);
    }

    // Apply dark mode class to HTML element for Tailwind
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('darkMode', isDarkMode);
  }, [isDarkMode]);

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  const handleTryDemo = () => {
    monitoring.trackDemoInteraction('custom_analysis', 'try_demo_clicked', {
      currentView,
      timestamp: new Date().toISOString()
    });
    setCurrentView('demo');
  };

  const handleProfessionalDemo = () => {
    monitoring.trackDemoInteraction('professional_scenarios', 'professional_demo_clicked', {
      currentView,
      timestamp: new Date().toISOString()
    });
    setCurrentView('scenarios');
  };

  // PIVOT: Maya Scenario Handler (MVP Sprint 1)
  const handleMayaScenario = () => {
    setCurrentView('educator-response');
  };

  const handleAnalysisComplete = (results) => {
    setAnalysisResults(results);
    setCurrentView('results');
  };

  const handleScenarioAnalyze = async (scenarioData) => {
    // Real API call to backend for scenario analysis (Milestone #108)
    const analysisStartTime = Date.now();

    // Track scenario analysis start
    monitoring.trackDemoInteraction(scenarioData.scenarioContext?.title || 'scenario_analysis', 'analysis_started', {
      scenarioType: scenarioData.scenarioContext?.category,
      transcriptLength: scenarioData.transcript?.length,
      hasMetadata: !!scenarioData.metadata
    });

    setIsAnalyzing(true);
    setAnalysisProgress({ status: 'submitting', message: 'Submitting scenario for analysis...', progress: 0 });

    try {
      console.log('Starting scenario analysis with real API...', scenarioData);

      // Submit transcript to analysis API
      const apiCallStartTime = Date.now();
      const response = await fetch(API_ENDPOINTS.ANALYZE_TRANSCRIPT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          transcript: scenarioData.transcript,
          metadata: scenarioData.metadata
        }),
      });
      const apiCallDuration = Date.now() - apiCallStartTime;

      // Track API call performance
      monitoring.trackDependency('ANALYZE_TRANSCRIPT', API_ENDPOINTS.ANALYZE_TRANSCRIPT, apiCallDuration, response.status, response.ok);
      monitoring.trackCostEvent('api', 'transcript_analysis_request', {
        responseTime: apiCallDuration,
        statusCode: response.status,
        dataSize: JSON.stringify({transcript: scenarioData.transcript, metadata: scenarioData.metadata}).length
      });

      const submitData = await response.json();

      if (!response.ok) {
        throw new Error(submitData.detail?.message || submitData.detail || 'Analysis submission failed');
      }

      console.log('Analysis submitted successfully:', submitData.analysis_id);
      setAnalysisProgress({ status: 'processing', message: 'Analysis submitted! Processing...', progress: 10 });

      // Poll for analysis results
      const analysisId = submitData.analysis_id;
      let attempts = 0;
      const maxAttempts = 60; // 2 minutes maximum wait

      while (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds

        const statusResponse = await fetch(API_ENDPOINTS.ANALYZE_STATUS(analysisId));
        const status = await statusResponse.json();

        console.log(`Analysis status (attempt ${attempts + 1}):`, status.status, status.progress || 'no progress');

        // Update progress based on server status
        if (status.progress !== undefined) {
          setAnalysisProgress({
            status: status.status,
            message: status.message || 'Processing...',
            progress: status.progress
          });
        } else {
          // Fallback progress estimation
          const estimatedProgress = Math.min(10 + (attempts * 3), 90);
          setAnalysisProgress({
            status: status.status,
            message: status.message || 'Analyzing scenario...',
            progress: estimatedProgress
          });
        }

        if (status.status === 'complete') {
          setAnalysisProgress({ status: 'complete', message: 'Getting results...', progress: 100 });

          // Get final results
          const resultsResponse = await fetch(API_ENDPOINTS.ANALYZE_RESULTS(analysisId));
          const results = await resultsResponse.json();

          console.log('Analysis completed successfully:', results);

          // Add scenario context for enhanced display
          results.scenarioContext = scenarioData.scenarioContext;
          results.original_transcript = scenarioData.transcript;

          // Track successful analysis completion
          const totalAnalysisTime = Date.now() - analysisStartTime;
          monitoring.trackMLAnalysis('scenario_analysis', totalAnalysisTime, true, {
            analysisId: analysisId,
            scenarioType: scenarioData.scenarioContext?.category,
            attempts: attempts + 1,
            hasResults: !!results
          });
          monitoring.trackDemoInteraction(scenarioData.scenarioContext?.title || 'scenario_analysis', 'analysis_completed', {
            duration: totalAnalysisTime,
            success: true,
            attempts: attempts + 1
          });

          setIsAnalyzing(false);
          setAnalysisProgress(null);
          handleAnalysisComplete(results);
          return;
        } else if (status.status === 'error') {
          throw new Error(status.message || 'Analysis failed on server');
        }

        attempts++;
      }

      throw new Error('Analysis timed out after 2 minutes');

    } catch (error) {
      console.error('Scenario analysis failed:', error);

      // Track analysis failure
      const totalAnalysisTime = Date.now() - analysisStartTime;
      monitoring.trackException(error, {
        function: 'handleScenarioAnalyze',
        scenarioType: scenarioData.scenarioContext?.category,
        duration: totalAnalysisTime,
        errorType: 'analysis_failure'
      });
      monitoring.trackMLAnalysis('scenario_analysis', totalAnalysisTime, false, {
        error: error.message,
        scenarioType: scenarioData.scenarioContext?.category
      });

      setIsAnalyzing(false);
      setAnalysisProgress(null);

      // Show user-friendly error message
      alert(`Analysis failed: ${error.message}. Please try again.`);
    }
  };

  const handleStartNew = () => {
    setAnalysisResults(null);
    setCurrentView('demo');
  };

  const handleStartNewScenario = () => {
    setAnalysisResults(null);
    setCurrentView('scenarios');
  };

  const handleStartNewEducatorResponse = () => {
    setEducatorResponseResults(null);
    setCurrentView('educator-response');
  };

  const handleBackHome = () => {
    setCurrentView('home');
    setAnalysisResults(null);
    setEducatorResponseResults(null);
  };

  // PIVOT: Educator Response Analysis Handler (MVP Sprint 1)
  const handleEducatorResponseAnalysis = async (responseData) => {
    setIsAnalyzing(true);
    setAnalysisProgress({ status: 'submitting', message: 'Submitting your response for analysis...', progress: 0 });

    try {
      console.log('Starting educator response analysis...', responseData);

      // Submit educator response to new analysis API
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/analyze/educator-response`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(responseData),
      });

      const submitData = await response.json();

      if (!response.ok) {
        throw new Error(submitData.detail?.message || submitData.detail || 'Response analysis failed');
      }

      console.log('Response analysis submitted:', submitData.analysis_id);
      setAnalysisProgress({ status: 'processing', message: 'Analyzing pedagogical quality...', progress: 10 });

      // Poll for analysis results
      const analysisId = submitData.analysis_id;
      let attempts = 0;
      const maxAttempts = 60; // 2 minutes maximum wait

      while (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds

        const statusResponse = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/analyze/educator-response/status/${analysisId}`);
        const status = await statusResponse.json();

        console.log(`Response analysis status (attempt ${attempts + 1}):`, status.status, status.progress || 'no progress');

        // Update progress based on server status
        if (status.progress !== undefined) {
          setAnalysisProgress({
            status: status.status,
            message: status.message || 'Processing...',
            progress: status.progress
          });
        } else {
          // Fallback progress estimation
          const estimatedProgress = Math.min(10 + (attempts * 3), 90);
          setAnalysisProgress({
            status: status.status,
            message: status.message || 'Analyzing your response...',
            progress: estimatedProgress
          });
        }

        if (status.status === 'complete') {
          setAnalysisProgress({ status: 'complete', message: 'Getting coaching feedback...', progress: 100 });

          // Get final results
          const resultsResponse = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/analyze/educator-response/results/${analysisId}`);
          const results = await resultsResponse.json();

          console.log('Response analysis completed:', results);

          // Store educator response results and navigate
          setEducatorResponseResults(results);
          setIsAnalyzing(false);
          setAnalysisProgress(null);
          setCurrentView('educator-response-results');
          return;
        } else if (status.status === 'error') {
          throw new Error(status.message || 'Response analysis failed on server');
        }

        attempts++;
      }

      throw new Error('Response analysis timed out after 2 minutes');

    } catch (error) {
      console.error('Educator response analysis failed:', error);
      setIsAnalyzing(false);
      setAnalysisProgress(null);

      // Show user-friendly error message
      alert(`Analysis failed: ${error.message}. Please try again.`);
    }
  };

  // Render different views based on current state
  if (currentView === 'recommendations') {
    return <MockRecommendationsTest />;
  }

  if (currentView === 'scenarios') {
    return (
      <ScenarioSelection
        onScenarioAnalyze={handleScenarioAnalyze}
        onBackToHome={handleBackHome}
        isAnalyzing={isAnalyzing}
        analysisProgress={analysisProgress}
      />
    );
  }

  // PIVOT: Educator Response Input View (MVP Sprint 1)
  if (currentView === 'educator-response') {
    return (
      <EducatorResponseInput
        onResponseAnalysis={handleEducatorResponseAnalysis}
        onBackToHome={handleBackHome}
        isAnalyzing={isAnalyzing}
        analysisProgress={analysisProgress}
      />
    );
  }

  if (currentView === 'demo') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <header className="bg-white shadow-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-6">
              <div className="flex items-center">
                <Brain className="h-8 w-8 text-indigo-600 mr-3" />
                <h1 className="text-2xl font-bold text-gray-900">Cultivate Learning ML MVP</h1>
              </div>
              <button
                onClick={handleBackHome}
                className="text-gray-500 hover:text-gray-900 px-4 py-2 rounded-lg transition-colors"
                aria-label="Back to Home"
              >
                ← Back to Home
              </button>
            </div>
          </div>
        </header>
        <div className="py-8">
          <TranscriptSubmission onAnalysisComplete={handleAnalysisComplete} />
        </div>
      </div>
    );
  }

  // PIVOT: Educator Response Results View (MVP Sprint 1)
  if (currentView === 'educator-response-results') {
    return (
      <EducatorResponseResults
        results={educatorResponseResults}
        onStartNew={handleStartNewEducatorResponse}
      />
    );
  }

  if (currentView === 'results') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <header className="bg-white shadow-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-6">
              <div className="flex items-center">
                <Brain className="h-8 w-8 text-indigo-600 mr-3" />
                <h1 className="text-2xl font-bold text-gray-900">Cultivate Learning ML MVP</h1>
              </div>
              <div className="flex space-x-4">
                <button
                  onClick={handleBackHome}
                  className="text-gray-500 hover:text-gray-900 px-4 py-2 rounded-lg transition-colors"
                >
                  ← Home
                </button>
              </div>
            </div>
          </div>
        </header>
        <div className="py-8">
          <AnalysisResults
            results={analysisResults}
            onStartNew={handleStartNew}
            onStartNewScenario={handleStartNewScenario}
          />
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen transition-all duration-300 ${
      isDarkMode
        ? 'bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-900'
        : 'bg-gradient-to-br from-blue-50 to-indigo-100'
    }`} style={{ scrollBehavior: 'smooth' }}>
      {/* Professional Sticky Header */}
      <header className={`fixed top-0 left-0 right-0 z-50 backdrop-blur-md shadow-sm border-b transition-all duration-300 ${
        isDarkMode
          ? 'bg-slate-900/95 border-slate-700/50'
          : 'bg-white/95 border-slate-200/50'
      }`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <div className="w-10 h-10 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center mr-3">
                <Sparkles className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className={`text-xl font-bold transition-colors ${
                  isDarkMode ? 'text-white' : 'text-slate-900'
                }`}>Cultivate</h1>
                <p className="text-xs text-indigo-600 font-medium">University of Washington</p>
              </div>
            </div>

            {/* Enhanced Navigation */}
            <nav className="hidden md:flex items-center space-x-8">
              {/* Dark mode toggle */}
              <button
                onClick={toggleDarkMode}
                className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors ${
                  isDarkMode
                    ? 'bg-slate-800 hover:bg-slate-700 text-yellow-400'
                    : 'bg-slate-100 hover:bg-slate-200 text-slate-600'
                }`}
                aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
              </button>

              <a
                href="#features"
                className={`relative font-medium group transition-colors ${
                  isDarkMode
                    ? 'text-slate-300 hover:text-indigo-400'
                    : 'text-slate-600 hover:text-indigo-600'
                }`}
                onClick={(e) => {
                  e.preventDefault();
                  document.getElementById('features').scrollIntoView({ behavior: 'smooth' });
                }}
              >
                Features
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-indigo-600 transition-all duration-300 group-hover:w-full"></span>
              </a>
              <a
                href="#demo"
                className={`relative font-medium group transition-colors ${
                  isDarkMode
                    ? 'text-slate-300 hover:text-indigo-400'
                    : 'text-slate-600 hover:text-indigo-600'
                }`}
                onClick={(e) => {
                  e.preventDefault();
                  document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
                }}
              >
                Demo
                <span className={`absolute -bottom-1 left-0 w-0 h-0.5 transition-all duration-300 group-hover:w-full ${
                  isDarkMode ? 'bg-indigo-400' : 'bg-indigo-600'
                }`}></span>
              </a>
              <a
                href="#research"
                className={`relative font-medium group transition-colors ${
                  isDarkMode
                    ? 'text-slate-300 hover:text-indigo-400'
                    : 'text-slate-600 hover:text-indigo-600'
                }`}
                onClick={(e) => {
                  e.preventDefault();
                  // Scroll to footer since we don't have a research section
                  document.querySelector('footer').scrollIntoView({ behavior: 'smooth' });
                }}
              >
                Research
                <span className={`absolute -bottom-1 left-0 w-0 h-0.5 transition-all duration-300 group-hover:w-full ${
                  isDarkMode ? 'bg-indigo-400' : 'bg-indigo-600'
                }`}></span>
              </a>

              {/* CTA Button in Header */}
              <button
                onClick={handleProfessionalDemo}
                className="ml-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-6 py-2.5 rounded-xl font-semibold text-sm hover:from-indigo-700 hover:to-purple-700 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105"
              >
                Try Demo
              </button>
            </nav>

            {/* Mobile menu button */}
            <div className="md:hidden flex items-center space-x-3">
              {/* Mobile dark mode toggle */}
              <button
                onClick={toggleDarkMode}
                className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors ${
                  isDarkMode
                    ? 'bg-slate-800 hover:bg-slate-700 text-yellow-400'
                    : 'bg-slate-100 hover:bg-slate-200 text-slate-600'
                }`}
                aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
              </button>

              <button className={`w-10 h-10 flex items-center justify-center rounded-lg transition-colors ${
                isDarkMode
                  ? 'text-slate-300 hover:bg-slate-800'
                  : 'text-slate-600 hover:bg-slate-100'
              }`}
                aria-label="Open mobile menu">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 overflow-hidden"
               data-testid="hero-section"
               style={{ paddingTop: '8rem' }}> {/* Account for fixed header */}
        {/* Elegant background gradient with subtle texture */}
        <div className="absolute inset-0 bg-gradient-to-br from-slate-50 via-blue-50/30 to-indigo-50/50"></div>
        <div className="absolute inset-0">
          <div className="absolute top-0 -left-4 w-96 h-96 bg-blue-300/20 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-pulse"></div>
          <div className="absolute top-0 -right-4 w-96 h-96 bg-indigo-300/20 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-pulse delay-1000"></div>
          <div className="absolute -bottom-8 left-20 w-96 h-96 bg-purple-300/20 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-pulse delay-2000"></div>
        </div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            {/* Premium badge */}
            <div className="inline-flex items-center px-4 py-2 rounded-full bg-white/90 backdrop-blur-sm border border-indigo-100/50 shadow-lg mb-8">
              <span className="w-2 h-2 bg-emerald-400 rounded-full mr-2 animate-pulse"></span>
              <span className="text-sm font-medium text-slate-700">Powered by Advanced Machine Learning</span>
            </div>

            <h1 className="text-4xl sm:text-5xl md:text-7xl lg:text-8xl font-black text-slate-900 mb-6 md:mb-8 leading-none px-4 sm:px-0">
              <span className="block">Transform</span>
              <span className="block bg-gradient-to-r from-indigo-600 via-purple-600 to-blue-600 bg-clip-text text-transparent">
                Early Learning
              </span>
            </h1>

            <p className="text-lg sm:text-xl md:text-2xl text-slate-600 mb-8 md:mb-12 max-w-4xl mx-auto leading-relaxed font-light px-4 sm:px-0">
              Intelligent analysis of educator-child interactions that reveals the science behind
              exceptional early childhood education—helping teachers unlock every child's potential.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 sm:gap-6 justify-center items-center px-4 sm:px-0">
              <button
                onClick={handleProfessionalDemo}
                className="group relative inline-flex items-center justify-center w-full sm:w-auto px-8 sm:px-10 py-4 bg-gradient-to-r from-indigo-600 to-blue-600 text-white rounded-2xl font-semibold text-lg shadow-xl hover:shadow-2xl transform hover:-translate-y-1 transition-all duration-300 hover:from-indigo-700 hover:to-blue-700"
              >
                <span className="absolute inset-0 rounded-2xl bg-gradient-to-r from-indigo-400 to-blue-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300 blur"></span>
                <span className="relative flex items-center">
                  <span className="hidden sm:inline">Experience the Demo</span>
                  <span className="sm:hidden">Try Demo</span>
                  <svg className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform duration-300" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </span>
              </button>

              <button
                onClick={handleTryDemo}
                className="group inline-flex items-center justify-center w-full sm:w-auto px-8 sm:px-10 py-4 bg-white/90 backdrop-blur-sm text-slate-700 rounded-2xl font-semibold text-lg border border-slate-200/50 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-300 hover:bg-white hover:border-slate-300/50"
              >
                <span className="hidden sm:inline">Try Your Own Data</span>
                <span className="sm:hidden">Your Data</span>
                <svg className="w-4 h-4 ml-2 group-hover:translate-x-0.5 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </button>
            </div>

            {/* Trust indicators */}
            <div className="mt-12 sm:mt-16 pt-6 sm:pt-8 border-t border-slate-200/50 px-4 sm:px-0">
              <p className="text-xs sm:text-sm font-medium text-slate-500 uppercase tracking-wider mb-4 sm:mb-6 text-center">Trusted by Leading Educational Institutions</p>
              <div className="flex flex-col sm:flex-row justify-center items-center space-y-2 sm:space-y-0 sm:space-x-8 lg:space-x-12 opacity-60">
                <div className="text-slate-400 font-semibold text-sm sm:text-base text-center">University of Washington</div>
                <div className="hidden sm:block w-1 h-1 bg-slate-300 rounded-full"></div>
                <div className="text-slate-400 font-semibold text-sm sm:text-base text-center">Early Learning Research</div>
                <div className="hidden sm:block w-1 h-1 bg-slate-300 rounded-full"></div>
                <div className="text-slate-400 font-semibold text-sm sm:text-base text-center">Constructivist Pedagogy</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative py-32 bg-gradient-to-b from-white to-slate-50/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16 sm:mb-20 px-4 sm:px-0">
            <div className="inline-block">
              <h2 className="text-xs sm:text-sm font-semibold text-indigo-600 tracking-wide uppercase mb-3 sm:mb-4">Advanced AI Capabilities</h2>
              <h3 className="text-3xl sm:text-4xl md:text-6xl font-bold text-slate-900 mb-4 sm:mb-6 leading-tight">
                Comprehensive Learning
                <span className="block text-indigo-600">Intelligence</span>
              </h3>
            </div>
            <p className="text-lg sm:text-xl text-slate-600 max-w-3xl mx-auto leading-relaxed px-4 sm:px-0">
              Our sophisticated AI platform analyzes every dimension of educational interactions,
              providing insights that transform how teachers connect with children.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8 px-4 sm:px-0">
            <PremiumFeatureCard
              icon={<Camera className="h-10 w-10" />}
              title="Vision Intelligence"
              description="Advanced computer vision analyzes body language, engagement cues, and visual interaction patterns to reveal the unspoken dynamics of learning."
              gradient="from-blue-500 to-cyan-600"
              isDarkMode={isDarkMode}
            />
            <PremiumFeatureCard
              icon={<Mic className="h-10 w-10" />}
              title="Conversation Analysis"
              description="Sophisticated audio processing captures tone, timing, and linguistic nuances that distinguish exceptional educators from the rest."
              gradient="from-purple-500 to-pink-600"
              isDarkMode={isDarkMode}
            />
            <PremiumFeatureCard
              icon={<Brain className="h-10 w-10" />}
              title="Language Understanding"
              description="Deep learning models trained on educational research extract pedagogical intent and measure constructivist alignment in real conversations."
              gradient="from-emerald-500 to-teal-600"
              isDarkMode={isDarkMode}
            />
            <PremiumFeatureCard
              icon={<Users className="h-10 w-10" />}
              title="Relationship Mapping"
              description="Identify and analyze the complex web of social-emotional interactions that form the foundation of effective early childhood education."
              gradient="from-orange-500 to-red-600"
              isDarkMode={isDarkMode}
            />
            <PremiumFeatureCard
              icon={<BarChart3 className="h-10 w-10" />}
              title="Predictive Insights"
              description="Machine learning models trained on longitudinal data predict learning outcomes and identify intervention opportunities before they're needed."
              gradient="from-indigo-500 to-purple-600"
              isDarkMode={isDarkMode}
            />
            <PremiumFeatureCard
              icon={<Heart className="h-10 w-10" />}
              title="Emotional Intelligence"
              description="Measure and enhance the quality of educator-child emotional connections that research shows are critical for lifelong learning success."
              gradient="from-pink-500 to-rose-600"
              isDarkMode={isDarkMode}
            />
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section id="demo" className="relative py-32 bg-slate-900 overflow-hidden">
        {/* Dark gradient background with subtle pattern */}
        <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-900"></div>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(120,119,198,0.3),transparent_50%)]"></div>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_80%,rgba(67,56,202,0.3),transparent_50%)]"></div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16 sm:mb-20 px-4 sm:px-0">
            <div className="inline-block">
              <h2 className="text-xs sm:text-sm font-semibold text-indigo-400 tracking-wide uppercase mb-3 sm:mb-4">Experience the Platform</h2>
              <h3 className="text-3xl sm:text-4xl md:text-6xl font-bold text-white mb-4 sm:mb-6 leading-tight">
                See Intelligence
                <span className="block text-transparent bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text">
                  In Action
                </span>
              </h3>
            </div>
            <p className="text-lg sm:text-xl text-slate-300 max-w-3xl mx-auto leading-relaxed px-4 sm:px-0">
              Choose your path to discovery—explore our curated educator scenarios or analyze your own interactions
              with our advanced AI models.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 sm:gap-8 mb-12 sm:mb-16 px-4 sm:px-0">
            {/* Professional Demo Card */}
            <div className="group relative">
              <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-3xl blur-lg opacity-20 group-hover:opacity-40 transition duration-1000"></div>
              <div className="relative bg-white/10 backdrop-blur-xl rounded-2xl p-8 border border-white/20 hover:bg-white/15 transition-all duration-300 transform hover:-translate-y-1">
                <div className="flex items-center justify-center w-16 h-16 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl mb-6 shadow-xl">
                  <Users className="h-10 w-10 text-white" />
                </div>

                <h3 className="text-2xl font-bold text-white mb-4">Curated Scenarios</h3>
                <p className="text-slate-300 mb-6 leading-relaxed text-lg">
                  Explore professionally crafted educator-child interactions spanning different quality levels,
                  age groups, and educational contexts.
                </p>

                <div className="space-y-3 mb-8">
                  <div className="flex items-center text-slate-300">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full mr-3"></div>
                    <span>8 realistic interaction scenarios</span>
                  </div>
                  <div className="flex items-center text-slate-300">
                    <div className="w-2 h-2 bg-blue-400 rounded-full mr-3"></div>
                    <span>Multiple age groups (2-8 years)</span>
                  </div>
                  <div className="flex items-center text-slate-300">
                    <div className="w-2 h-2 bg-purple-400 rounded-full mr-3"></div>
                    <span>Various teaching quality levels</span>
                  </div>
                </div>

                <button
                  onClick={handleProfessionalDemo}
                  className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-4 px-6 rounded-xl font-semibold text-lg hover:from-indigo-700 hover:to-purple-700 transition-all duration-300 shadow-xl hover:shadow-2xl transform hover:scale-105"
                >
                  Explore Scenarios
                </button>
              </div>
            </div>

            {/* PIVOT: Maya Scenario Response Coaching Card (MVP Sprint 1) */}
            <div className="group relative">
              <div className="absolute -inset-1 bg-gradient-to-r from-orange-500 to-red-600 rounded-3xl blur-lg opacity-20 group-hover:opacity-40 transition duration-1000"></div>
              <div className="relative bg-white/10 backdrop-blur-xl rounded-2xl p-8 border border-white/20 hover:bg-white/15 transition-all duration-300 transform hover:-translate-y-1">
                <div className="flex items-center justify-center w-16 h-16 bg-gradient-to-r from-orange-500 to-red-600 rounded-2xl mb-6 shadow-xl">
                  <Brain className="h-10 w-10 text-white" />
                </div>

                <h3 className="text-2xl font-bold text-white mb-4">AI Coaching Demo</h3>
                <p className="text-slate-300 mb-6 leading-relaxed text-lg">
                  Type your response to Maya's puzzle frustration and receive personalized AI coaching
                  based on evidence-based early childhood pedagogy.
                </p>

                <div className="space-y-3 mb-8">
                  <div className="flex items-center text-slate-300">
                    <div className="w-2 h-2 bg-orange-400 rounded-full mr-3"></div>
                    <span>Official UW demo scenario</span>
                  </div>
                  <div className="flex items-center text-slate-300">
                    <div className="w-2 h-2 bg-red-400 rounded-full mr-3"></div>
                    <span>5-category coaching feedback</span>
                  </div>
                  <div className="flex items-center text-slate-300">
                    <div className="w-2 h-2 bg-yellow-400 rounded-full mr-3"></div>
                    <span>Evidence-based recommendations</span>
                  </div>
                </div>

                <button
                  onClick={handleMayaScenario}
                  className="w-full bg-gradient-to-r from-orange-600 to-red-600 text-white py-4 px-6 rounded-xl font-semibold text-lg hover:from-orange-700 hover:to-red-700 transition-all duration-300 shadow-xl hover:shadow-2xl transform hover:scale-105"
                >
                  Try Maya Scenario
                </button>
              </div>
            </div>

            {/* Custom Analysis Card */}
            <div className="group relative">
              <div className="absolute -inset-1 bg-gradient-to-r from-emerald-500 to-blue-600 rounded-3xl blur-lg opacity-20 group-hover:opacity-40 transition duration-1000"></div>
              <div className="relative bg-white/10 backdrop-blur-xl rounded-2xl p-8 border border-white/20 hover:bg-white/15 transition-all duration-300 transform hover:-translate-y-1">
                <div className="flex items-center justify-center w-16 h-16 bg-gradient-to-r from-emerald-500 to-blue-600 rounded-2xl mb-6 shadow-xl">
                  <Camera className="h-10 w-10 text-white" />
                </div>

                <h3 className="text-2xl font-bold text-white mb-4">Your Own Data</h3>
                <p className="text-slate-300 mb-6 leading-relaxed text-lg">
                  Upload your own educator-child interaction transcripts and receive personalized AI analysis
                  with actionable insights.
                </p>

                <div className="space-y-3 mb-8">
                  <div className="flex items-center text-slate-300">
                    <div className="w-2 h-2 bg-emerald-400 rounded-full mr-3"></div>
                    <span>Real-time analysis engine</span>
                  </div>
                  <div className="flex items-center text-slate-300">
                    <div className="w-2 h-2 bg-blue-400 rounded-full mr-3"></div>
                    <span>Personalized recommendations</span>
                  </div>
                  <div className="flex items-center text-slate-300">
                    <div className="w-2 h-2 bg-cyan-400 rounded-full mr-3"></div>
                    <span>Research-backed insights</span>
                  </div>
                </div>

                <button
                  onClick={handleTryDemo}
                  className="w-full bg-white/20 backdrop-blur-sm text-white py-4 px-6 rounded-xl font-semibold text-lg border border-white/30 hover:bg-white/30 hover:border-white/50 transition-all duration-300 shadow-xl hover:shadow-2xl transform hover:scale-105"
                >
                  Try Custom Analysis
                </button>
              </div>
            </div>
          </div>

          {/* Sample Results Preview */}
          <div className="bg-white/5 backdrop-blur-xl rounded-3xl p-8 border border-white/20">
            <div className="text-center mb-8">
              <h4 className="text-2xl font-bold text-white mb-2">Sample Analysis Results</h4>
              <p className="text-slate-300">See the kind of insights you'll receive from our AI analysis</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-gradient-to-br from-emerald-500/20 to-teal-600/20 backdrop-blur-sm rounded-2xl p-6 border border-emerald-500/30">
                <div className="flex items-center mb-4">
                  <div className="w-4 h-4 bg-emerald-400 rounded-full mr-3"></div>
                  <span className="font-bold text-emerald-300 text-lg">Pedagogical Quality</span>
                </div>
                <div className="text-3xl font-bold text-white mb-2">8.7/10</div>
                <p className="text-emerald-200 text-sm">Strong constructivist alignment with research-backed questioning techniques</p>
              </div>

              <div className="bg-gradient-to-br from-blue-500/20 to-cyan-600/20 backdrop-blur-sm rounded-2xl p-6 border border-blue-500/30">
                <div className="flex items-center mb-4">
                  <div className="w-4 h-4 bg-blue-400 rounded-full mr-3"></div>
                  <span className="font-bold text-blue-300 text-lg">Engagement Level</span>
                </div>
                <div className="text-3xl font-bold text-white mb-2">High</div>
                <p className="text-blue-200 text-sm">Active participation patterns and sustained attention indicators detected</p>
              </div>

              <div className="bg-gradient-to-br from-purple-500/20 to-pink-600/20 backdrop-blur-sm rounded-2xl p-6 border border-purple-500/30">
                <div className="flex items-center mb-4">
                  <div className="w-4 h-4 bg-purple-400 rounded-full mr-3"></div>
                  <span className="font-bold text-purple-300 text-lg">Learning Moments</span>
                </div>
                <div className="text-3xl font-bold text-white mb-2">12</div>
                <p className="text-purple-200 text-sm">Significant learning opportunities identified and analyzed for impact</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials & Social Proof Section */}
      <section className="py-24 bg-gradient-to-r from-slate-50 to-blue-50/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12 sm:mb-16 px-4 sm:px-0">
            <h2 className="text-xs sm:text-sm font-semibold text-indigo-600 tracking-wide uppercase mb-3 sm:mb-4">Trusted by Educators</h2>
            <h3 className="text-2xl sm:text-3xl md:text-4xl font-bold text-slate-900 mb-4 sm:mb-6">
              Transforming Early Learning Nationwide
            </h3>
            <p className="text-lg sm:text-xl text-slate-600 max-w-3xl mx-auto">
              Our research-backed platform is helping educators and institutions create more meaningful learning experiences.
            </p>
          </div>

          {/* Testimonials Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8 mb-12 sm:mb-16 px-4 sm:px-0">
            <div className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-shadow duration-300">
              <div className="flex items-center mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-teal-600 rounded-full flex items-center justify-center text-white font-bold text-lg mr-4">
                  M
                </div>
                <div>
                  <h4 className="font-semibold text-slate-900">Dr. Maria Chen</h4>
                  <p className="text-slate-600 text-sm">Early Learning Specialist</p>
                </div>
              </div>
              <p className="text-slate-700 leading-relaxed">
                "The AI insights have completely changed how I understand my interactions with children.
                I can see patterns I never noticed before."
              </p>
              <div className="mt-4 flex text-yellow-400">
                {[...Array(5)].map((_, i) => (
                  <svg key={i} className="w-4 h-4 fill-current" viewBox="0 0 20 20">
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
                  </svg>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-shadow duration-300">
              <div className="flex items-center mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full flex items-center justify-center text-white font-bold text-lg mr-4">
                  J
                </div>
                <div>
                  <h4 className="font-semibold text-slate-900">James Rodriguez</h4>
                  <p className="text-slate-600 text-sm">Preschool Director</p>
                </div>
              </div>
              <p className="text-slate-700 leading-relaxed">
                "This platform helps our teachers understand the quality of their interactions in real-time.
                It's like having a research assistant for every conversation."
              </p>
              <div className="mt-4 flex text-yellow-400">
                {[...Array(5)].map((_, i) => (
                  <svg key={i} className="w-4 h-4 fill-current" viewBox="0 0 20 20">
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
                  </svg>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-shadow duration-300">
              <div className="flex items-center mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-600 rounded-full flex items-center justify-center text-white font-bold text-lg mr-4">
                  S
                </div>
                <div>
                  <h4 className="font-semibold text-slate-900">Sarah Johnson</h4>
                  <p className="text-slate-600 text-sm">Head Start Teacher</p>
                </div>
              </div>
              <p className="text-slate-700 leading-relaxed">
                "The constructivist alignment scores give me confidence that I'm supporting
                children's natural learning processes effectively."
              </p>
              <div className="mt-4 flex text-yellow-400">
                {[...Array(5)].map((_, i) => (
                  <svg key={i} className="w-4 h-4 fill-current" viewBox="0 0 20 20">
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
                  </svg>
                ))}
              </div>
            </div>
          </div>

          {/* Research Impact Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 sm:gap-8 text-center px-4 sm:px-0">
            <div className="bg-white rounded-xl p-6 shadow-lg">
              <div className="text-3xl font-bold text-indigo-600 mb-2">500+</div>
              <div className="text-slate-600 font-medium">Educators Trained</div>
            </div>
            <div className="bg-white rounded-xl p-6 shadow-lg">
              <div className="text-3xl font-bold text-emerald-600 mb-2">2,400+</div>
              <div className="text-slate-600 font-medium">Interactions Analyzed</div>
            </div>
            <div className="bg-white rounded-xl p-6 shadow-lg">
              <div className="text-3xl font-bold text-blue-600 mb-2">18%</div>
              <div className="text-slate-600 font-medium">Average Quality Improvement</div>
            </div>
            <div className="bg-white rounded-xl p-6 shadow-lg">
              <div className="text-3xl font-bold text-purple-600 mb-2">95%</div>
              <div className="text-slate-600 font-medium">User Satisfaction</div>
            </div>
          </div>
        </div>
      </section>

      {/* Professional Footer */}
      <footer className="relative bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white overflow-hidden">
        {/* Background pattern */}
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_80%,rgba(120,119,198,0.2),transparent_50%)]"></div>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_20%,rgba(67,56,202,0.2),transparent_50%)]"></div>

        <div className="relative">
          {/* Main footer content */}
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12">

              {/* Company info */}
              <div className="lg:col-span-2">
                <div className="flex items-center mb-6">
                  <div className="w-10 h-10 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center mr-4">
                    <Sparkles className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold">Cultivate</h3>
                    <p className="text-indigo-400 text-sm font-medium">University of Washington</p>
                  </div>
                </div>
                <p className="text-slate-300 text-lg leading-relaxed mb-6 max-w-md">
                  Transforming early childhood education through advanced AI analysis and research-backed pedagogical insights.
                </p>
                <div className="flex space-x-4">
                  <a href="#" className="w-12 h-12 bg-white/10 rounded-xl flex items-center justify-center hover:bg-white/20 transition-colors group">
                    <svg className="w-6 h-6 text-slate-300 group-hover:text-white transition-colors" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M24 4.557c-.883.392-1.832.656-2.828.775 1.017-.609 1.798-1.574 2.165-2.724-.951.564-2.005.974-3.127 1.195-.897-.957-2.178-1.555-3.594-1.555-3.179 0-5.515 2.966-4.797 6.045-4.091-.205-7.719-2.165-10.148-5.144-1.29 2.213-.669 5.108 1.523 6.574-.806-.026-1.566-.247-2.229-.616-.054 2.281 1.581 4.415 3.949 4.89-.693.188-1.452.232-2.224.084.626 1.956 2.444 3.379 4.6 3.419-2.07 1.623-4.678 2.348-7.29 2.04 2.179 1.397 4.768 2.212 7.548 2.212 9.142 0 14.307-7.721 13.995-14.646.962-.695 1.797-1.562 2.457-2.549z"/>
                    </svg>
                  </a>
                  <a href="#" className="w-12 h-12 bg-white/10 rounded-xl flex items-center justify-center hover:bg-white/20 transition-colors group">
                    <svg className="w-6 h-6 text-slate-300 group-hover:text-white transition-colors" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                    </svg>
                  </a>
                  <a href="#" className="w-12 h-12 bg-white/10 rounded-xl flex items-center justify-center hover:bg-white/20 transition-colors group">
                    <svg className="w-6 h-6 text-slate-300 group-hover:text-white transition-colors" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                    </svg>
                  </a>
                </div>
              </div>

              {/* Research & Education */}
              <div>
                <h4 className="text-lg font-bold text-white mb-6">Research</h4>
                <ul className="space-y-4">
                  <li><a href="#" className="text-slate-300 hover:text-indigo-400 transition-colors">Our Methodology</a></li>
                  <li><a href="#" className="text-slate-300 hover:text-indigo-400 transition-colors">Published Papers</a></li>
                  <li><a href="#" className="text-slate-300 hover:text-indigo-400 transition-colors">Constructivist Theory</a></li>
                  <li><a href="#" className="text-slate-300 hover:text-indigo-400 transition-colors">Data Ethics</a></li>
                  <li><a href="#" className="text-slate-300 hover:text-indigo-400 transition-colors">Academic Partnerships</a></li>
                </ul>
              </div>

              {/* Platform */}
              <div>
                <h4 className="text-lg font-bold text-white mb-6">Platform</h4>
                <ul className="space-y-4">
                  <li><a href="#demo" className="text-slate-300 hover:text-indigo-400 transition-colors">Try Demo</a></li>
                  <li><a href="#features" className="text-slate-300 hover:text-indigo-400 transition-colors">AI Capabilities</a></li>
                  <li><a href="#" className="text-slate-300 hover:text-indigo-400 transition-colors">API Documentation</a></li>
                  <li><a href="#" className="text-slate-300 hover:text-indigo-400 transition-colors">Privacy Policy</a></li>
                  <li><a href="#" className="text-slate-300 hover:text-indigo-400 transition-colors">Terms of Service</a></li>
                </ul>
              </div>
            </div>
          </div>

          {/* Bottom section */}
          <div className="border-t border-white/10">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
              <div className="md:flex md:items-center md:justify-between">
                <div className="flex flex-col md:flex-row md:items-center md:space-x-8">
                  <p className="text-slate-400 text-sm">
                    © 2024 University of Washington - Cultivate Research Project. All rights reserved.
                  </p>
                  <div className="flex items-center mt-4 md:mt-0 space-x-6">
                    <a href="#" className="text-slate-400 hover:text-indigo-400 text-sm transition-colors">Privacy</a>
                    <a href="#" className="text-slate-400 hover:text-indigo-400 text-sm transition-colors">Terms</a>
                    <a href="#" className="text-slate-400 hover:text-indigo-400 text-sm transition-colors">Contact</a>
                  </div>
                </div>
                <div className="mt-4 md:mt-0">
                  <p className="text-slate-500 text-xs">
                    Funded by NSF Grant #1122334 • Early Learning Research Initiative
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

const PremiumFeatureCard = React.memo(function PremiumFeatureCard({ icon, title, description, gradient, isDarkMode = false }) {
  return (
    <div className="group relative">
      {/* Glow effect */}
      <div className={`absolute -inset-1 bg-gradient-to-r ${gradient} rounded-3xl blur-lg opacity-0 group-hover:opacity-20 transition duration-1000 group-hover:duration-200`}></div>

      {/* Card */}
      <div className={`relative rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 ${
        isDarkMode
          ? 'bg-slate-800/90 border border-slate-700/60'
          : 'bg-white border border-slate-200/60'
      }`}>
        {/* Icon with gradient background */}
        <div className={`inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r ${gradient} rounded-2xl mb-6 shadow-lg`}>
          <div className="text-white">
            {icon}
          </div>
        </div>

        <h3 className={`text-2xl font-bold mb-4 transition-colors ${
          isDarkMode
            ? 'text-white group-hover:text-slate-200'
            : 'text-slate-900 group-hover:text-slate-700'
        }`}>
          {title}
        </h3>

        <p className={`leading-relaxed text-lg transition-colors ${
          isDarkMode ? 'text-slate-300' : 'text-slate-600'
        }`}>
          {description}
        </p>

        {/* Subtle hover indicator */}
        <div className={`mt-6 flex items-center transition-colors ${
          isDarkMode
            ? 'text-slate-500 group-hover:text-slate-300'
            : 'text-slate-400 group-hover:text-slate-600'
        }`}>
          <span className="text-sm font-medium">Learn more</span>
          <svg className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </div>
      </div>
    </div>
  );
});

export default App;