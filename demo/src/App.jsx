import React, { useState } from 'react';
import { Brain, Users, Camera, Mic, BarChart3, Heart } from 'lucide-react';
import TranscriptSubmission from './components/TranscriptSubmission';
import AnalysisResults from './components/AnalysisResults';
import MockAnalysisTest from './components/MockAnalysisTest';
import MockRecommendationsTest from './components/MockRecommendationsTest';
import ScenarioSelection from './components/ScenarioSelection';

function App() {
  const [currentView, setCurrentView] = useState('home'); // 'home', 'demo', 'scenarios', 'results', 'recommendations'
  const [analysisResults, setAnalysisResults] = useState(null);

  const handleTryDemo = () => {
    setCurrentView('demo');
  };

  const handleProfessionalDemo = () => {
    setCurrentView('scenarios');
  };

  const handleAnalysisComplete = (results) => {
    setAnalysisResults(results);
    setCurrentView('results');
  };

  const handleScenarioAnalyze = async (scenarioData) => {
    // Simulate API call with scenario data
    // In production, this would call the actual transcript analysis API
    try {
      const mockResults = {
        analysis_id: 'scenario-' + Date.now(),
        status: 'complete',
        transcript_summary: {
          conversational_turns: scenarioData.transcript.split('\n\n').length,
          word_count: scenarioData.transcript.split(' ').length,
          estimated_duration_minutes: scenarioData.metadata?.duration_minutes || 3
        },
        ml_predictions: {
          question_quality: scenarioData.scenarioContext.expectedQuality === 'exemplary' ? 0.85 :
                            scenarioData.scenarioContext.expectedQuality === 'proficient' ? 0.72 :
                            scenarioData.scenarioContext.expectedQuality === 'developing' ? 0.58 : 0.42,
          wait_time_appropriate: scenarioData.scenarioContext.focusAreas.includes('wait-time') ? 0.78 : 0.65,
          scaffolding_present: scenarioData.scenarioContext.focusAreas.includes('scaffolding') ? 0.82 : 0.68,
          open_ended_questions: scenarioData.scenarioContext.focusAreas.includes('questioning') ? 0.79 : 0.61
        },
        class_scores: {
          emotional_support: scenarioData.scenarioContext.focusAreas.includes('emotional-support') ? 4.8 :
                            scenarioData.scenarioContext.expectedQuality === 'exemplary' ? 4.5 : 3.8,
          classroom_organization: 4.2,
          instructional_support: scenarioData.scenarioContext.expectedQuality === 'exemplary' ? 4.7 :
                                 scenarioData.scenarioContext.expectedQuality === 'proficient' ? 3.9 : 3.2,
          overall_score: scenarioData.scenarioContext.expectedQuality === 'exemplary' ? 4.5 :
                        scenarioData.scenarioContext.expectedQuality === 'proficient' ? 3.97 : 3.4
        },
        scaffolding_analysis: {
          total_scaffolding_instances: Math.floor(Math.random() * 8) + 3,
          scaffolding_effectiveness: scenarioData.scenarioContext.focusAreas.includes('scaffolding') ? 0.85 : 0.72
        },
        recommendations: [
          "Continue using excellent open-ended questioning techniques",
          "Consider providing slightly longer wait times for complex questions",
          "Strong emotional support evident throughout the interaction"
        ],
        enhanced_recommendations: [], // Will be populated by the recommendation engine
        processing_time: 2.5,
        created_at: new Date().toISOString(),
        completed_at: new Date().toISOString(),
        original_transcript: scenarioData.transcript,
        scenarioContext: scenarioData.scenarioContext // Pass scenario context for enhanced display
      };

      // Simulate processing delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      handleAnalysisComplete(mockResults);
    } catch (error) {
      console.error('Scenario analysis failed:', error);
      // Handle error state
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

  const handleBackHome = () => {
    setCurrentView('home');
    setAnalysisResults(null);
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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <Brain className="h-8 w-8 text-indigo-600 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900">Cultivate Learning ML MVP</h1>
            </div>
            <nav className="hidden md:flex space-x-8">
              <a href="#features" className="text-gray-500 hover:text-gray-900">Features</a>
              <a href="#demo" className="text-gray-500 hover:text-gray-900">Demo</a>
              <a href="#research" className="text-gray-500 hover:text-gray-900">Research</a>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
              AI-Powered Educator
              <span className="text-indigo-600"> Interaction Analysis</span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Analyzing educator-child interactions in early childhood education using cutting-edge
              machine learning to improve constructivist learning outcomes.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={handleProfessionalDemo}
                className="bg-indigo-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-indigo-700 transition-colors"
              >
                Professional Demo
              </button>
              <button
                onClick={handleTryDemo}
                className="bg-white text-indigo-600 border-2 border-indigo-600 px-8 py-3 rounded-lg font-semibold hover:bg-indigo-50 transition-colors"
              >
                Custom Analysis
              </button>
              <button className="border border-gray-300 text-gray-700 px-8 py-3 rounded-lg font-semibold hover:bg-gray-50 transition-colors">
                View Research
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">ML-Powered Analysis</h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Our MVP leverages multiple AI technologies to provide comprehensive insights
              into educator-child interactions.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <FeatureCard
              icon={<Camera className="h-8 w-8" />}
              title="Computer Vision"
              description="Analyze visual interactions, body language, and engagement patterns through video processing."
            />
            <FeatureCard
              icon={<Mic className="h-8 w-8" />}
              title="Audio Analysis"
              description="Process speech patterns, tone, and linguistic structures in educator-child conversations."
            />
            <FeatureCard
              icon={<Brain className="h-8 w-8" />}
              title="NLP Processing"
              description="Extract meaning and educational intent from transcribed conversations and interactions."
            />
            <FeatureCard
              icon={<Users className="h-8 w-8" />}
              title="Interaction Mapping"
              description="Identify and categorize different types of educational interactions and learning moments."
            />
            <FeatureCard
              icon={<BarChart3 className="h-8 w-8" />}
              title="Outcome Prediction"
              description="Predict learning outcomes based on interaction patterns and educator behaviors."
            />
            <FeatureCard
              icon={<Heart className="h-8 w-8" />}
              title="Constructivist Insights"
              description="Measure alignment with constructivist learning principles and provide actionable feedback."
            />
          </div>
        </div>
      </section>

      {/* Demo Section */}
      <section id="demo" className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Interactive Demo Options</h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Choose between our professional scenario collection or upload your own educator-child interaction data.
            </p>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Professional Demo</h3>
                <div className="border-2 border-solid border-indigo-200 rounded-lg p-6 text-center bg-indigo-50">
                  <Users className="h-12 w-12 text-indigo-600 mx-auto mb-4" />
                  <p className="text-gray-700 mb-2 font-medium">Curated Educator Scenarios</p>
                  <p className="text-sm text-gray-600 mb-4">
                    Professional scenarios across quality levels and age groups
                  </p>
                  <button
                    onClick={handleProfessionalDemo}
                    className="bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700 transition-colors"
                  >
                    Browse Scenarios
                  </button>
                </div>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Custom Analysis</h3>
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                  <Camera className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600 mb-2">Upload your own transcripts</p>
                  <p className="text-sm text-gray-500 mb-4">Paste educator-child interaction text</p>
                  <button
                    onClick={handleTryDemo}
                    className="bg-white text-indigo-600 border-2 border-indigo-600 px-6 py-2 rounded-lg hover:bg-indigo-50 transition-colors"
                  >
                    Try Custom Analysis
                  </button>
                </div>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Analysis Results</h3>
                <div className="space-y-4">
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <div className="flex items-center">
                      <div className="h-3 w-3 bg-green-500 rounded-full mr-3"></div>
                      <span className="font-semibold text-green-800">Constructivist Score: 8.7/10</span>
                    </div>
                    <p className="text-green-700 text-sm mt-1">Strong alignment with constructivist principles</p>
                  </div>

                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <div className="flex items-center">
                      <div className="h-3 w-3 bg-blue-500 rounded-full mr-3"></div>
                      <span className="font-semibold text-blue-800">Engagement Level: High</span>
                    </div>
                    <p className="text-blue-700 text-sm mt-1">Active participation detected</p>
                  </div>

                  <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                    <div className="flex items-center">
                      <div className="h-3 w-3 bg-purple-500 rounded-full mr-3"></div>
                      <span className="font-semibold text-purple-800">Learning Indicators: 12 detected</span>
                    </div>
                    <p className="text-purple-700 text-sm mt-1">Multiple learning moments identified</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <div className="flex items-center mb-4">
                <Brain className="h-6 w-6 text-indigo-400 mr-2" />
                <span className="font-semibold">Cultivate Learning ML</span>
              </div>
              <p className="text-gray-400">
                Advancing early childhood education through AI-powered interaction analysis.
              </p>
            </div>

            <div>
              <h3 className="font-semibold mb-4">Research</h3>
              <ul className="space-y-2 text-gray-400">
                <li>Constructivist Learning</li>
                <li>Educator Training</li>
                <li>ML in Education</li>
                <li>Interaction Analysis</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold mb-4">Technology</h3>
              <ul className="space-y-2 text-gray-400">
                <li>PyTorch & TensorFlow</li>
                <li>Computer Vision</li>
                <li>Natural Language Processing</li>
                <li>FastAPI & React</li>
              </ul>
            </div>
          </div>

          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2025 Cultivate Learning ML MVP. Built with Warren & Claude.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, description }) {
  return (
    <div className="bg-gray-50 rounded-xl p-6 hover:bg-gray-100 transition-colors">
      <div className="text-indigo-600 mb-4">
        {icon}
      </div>
      <h3 className="text-xl font-semibold text-gray-900 mb-2">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </div>
  );
}

export default App;