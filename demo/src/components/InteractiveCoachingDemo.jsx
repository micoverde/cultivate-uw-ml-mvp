import React, { useState } from 'react';
import { ArrowLeft, Send, BookOpen, Heart, Brain, Sparkles } from 'lucide-react';

const InteractiveCoachingDemo = ({ onBackToHome }) => {
  const [currentStep, setCurrentStep] = useState('scene'); // 'scene', 'response', 'feedback'
  const [userResponse, setUserResponse] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [feedback, setFeedback] = useState(null);

  // Sample scenario - reading a book about flowers
  const scenario = {
    title: "Reading Together: A Book About Flowers",
    setting: "Library Corner",
    childAge: "4 years old",
    context: "You and Emma are looking at a picture book about different types of flowers. Emma seems curious but is having trouble focusing.",
    childStatement: "Look! This flower is really pretty! But... but I don't know what it's called. Is it like the ones in my grandma's garden?",
    sceneDescription: "Emma (4) sits cross-legged on a colorful reading mat, pointing excitedly at a large picture book showing a bright sunflower. Her eyes are wide with curiosity, and she's leaning forward toward you with anticipation.",
    optimalResponse: "That's a beautiful sunflower, Emma! It does look like it could be in your grandma's garden. What do you notice about this sunflower that reminds you of the flowers there?",
    focusAreas: ["open-ended questioning", "building connections", "vocabulary development", "sustained engagement"]
  };

  const handleSubmitResponse = async () => {
    if (!userResponse.trim()) return;

    setIsAnalyzing(true);
    setCurrentStep('feedback');

    // Simulate ML analysis
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Generate coaching feedback based on response
    const analysis = analyzeResponse(userResponse);
    setFeedback(analysis);
    setIsAnalyzing(false);
  };

  const analyzeResponse = (response) => {
    const lowerResponse = response.toLowerCase();

    // Simple analysis based on key teaching principles
    const hasOpenQuestion = /\?/.test(response) && (lowerResponse.includes('what') || lowerResponse.includes('how') || lowerResponse.includes('why'));
    const acknowledgesChild = lowerResponse.includes('emma') || lowerResponse.includes('yes') || lowerResponse.includes('i see');
    const buildsConnection = lowerResponse.includes('grandma') || lowerResponse.includes('garden') || lowerResponse.includes('remember');
    const providesInfo = lowerResponse.includes('sunflower') || lowerResponse.includes('flower') || lowerResponse.includes('yellow');
    const encourages = lowerResponse.includes('great') || lowerResponse.includes('good') || lowerResponse.includes('wonderful');

    let score = 0;
    const strengths = [];
    const improvements = [];

    if (hasOpenQuestion) {
      score += 25;
      strengths.push("Uses open-ended questioning to extend thinking");
    } else {
      improvements.push("Try asking open-ended questions (what, how, why) to encourage deeper thinking");
    }

    if (acknowledgesChild) {
      score += 20;
      strengths.push("Acknowledges the child's observation and enthusiasm");
    } else {
      improvements.push("Acknowledge Emma's observation to validate her engagement");
    }

    if (buildsConnection) {
      score += 25;
      strengths.push("Makes meaningful connections to child's prior experience");
    } else {
      improvements.push("Connect to Emma's mention of grandma's garden to build on her knowledge");
    }

    if (providesInfo) {
      score += 20;
      strengths.push("Provides vocabulary and factual information");
    } else {
      improvements.push("Consider introducing the word 'sunflower' to build vocabulary");
    }

    if (encourages) {
      score += 10;
      strengths.push("Uses encouraging language");
    }

    // Determine overall quality
    let quality = 'developing';
    let qualityColor = 'text-orange-600';
    let qualityBg = 'bg-orange-100';

    if (score >= 80) {
      quality = 'exemplary';
      qualityColor = 'text-emerald-600';
      qualityBg = 'bg-emerald-100';
    } else if (score >= 60) {
      quality = 'proficient';
      qualityColor = 'text-blue-600';
      qualityBg = 'bg-blue-100';
    }

    return {
      score,
      quality,
      qualityColor,
      qualityBg,
      strengths,
      improvements,
      researchNote: "This analysis is based on CLASS (Classroom Assessment Scoring System) dimensions of Instructional Support and Emotional Support."
    };
  };

  const restartDemo = () => {
    setCurrentStep('scene');
    setUserResponse('');
    setFeedback(null);
    setIsAnalyzing(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white/95 backdrop-blur-md shadow-sm border-b border-slate-200/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <div className="w-10 h-10 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center mr-3">
                <Sparkles className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">Interactive Coaching Demo</h1>
                <p className="text-xs text-indigo-600 font-medium">Real-time AI Feedback</p>
              </div>
            </div>
            <button
              onClick={onBackToHome}
              className="text-slate-500 hover:text-slate-900 px-4 py-2 rounded-lg transition-colors hover:bg-slate-100"
              aria-label="Back to Home"
            >
              ← Back to Home
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

        {/* Scene Visualization Step */}
        {currentStep === 'scene' && (
          <div className="space-y-8">
            {/* Scene Header */}
            <div className="text-center">
              <div className="inline-flex items-center px-4 py-2 rounded-full bg-indigo-100 text-indigo-800 text-sm font-medium mb-4">
                <BookOpen className="w-4 h-4 mr-2" />
                Interactive Learning Scenario
              </div>
              <h2 className="text-3xl font-bold text-slate-900 mb-2">{scenario.title}</h2>
              <p className="text-lg text-slate-600">Age: {scenario.childAge} • Setting: {scenario.setting}</p>
            </div>

            {/* Visual Scene */}
            <div className="bg-white rounded-3xl shadow-xl overflow-hidden">
              <div className="relative h-96 bg-gradient-to-br from-amber-100 via-orange-50 to-pink-100">
                {/* Illustrated Scene */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center max-w-2xl px-8">
                    <div className="w-32 h-32 mx-auto mb-6 rounded-full bg-gradient-to-br from-pink-200 to-purple-200 flex items-center justify-center border-4 border-white shadow-lg">
                      <div className="text-6xl">👧🏽</div>
                    </div>
                    <div className="bg-white/90 backdrop-blur-sm rounded-2xl p-6 shadow-lg">
                      <h3 className="font-bold text-slate-900 mb-2">Emma is excited about the book</h3>
                      <p className="text-slate-700 leading-relaxed">{scenario.sceneDescription}</p>
                    </div>
                  </div>
                </div>

                {/* Decorative elements */}
                <div className="absolute top-4 left-4 w-16 h-16 bg-yellow-300 rounded-full opacity-70 animate-pulse">
                  <div className="text-3xl absolute inset-0 flex items-center justify-center">🌻</div>
                </div>
                <div className="absolute bottom-6 right-6 w-12 h-12 bg-green-300 rounded-full opacity-70">
                  <div className="text-2xl absolute inset-0 flex items-center justify-center">📚</div>
                </div>
                <div className="absolute top-1/2 left-8 w-8 h-8 bg-pink-300 rounded-full opacity-50"></div>
                <div className="absolute top-1/4 right-12 w-6 h-6 bg-purple-300 rounded-full opacity-50"></div>
              </div>

              {/* Context & Child's Statement */}
              <div className="p-8">
                <div className="grid md:grid-cols-2 gap-8">
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-slate-900 flex items-center">
                      <Heart className="w-5 h-5 text-pink-500 mr-2" />
                      Context
                    </h3>
                    <p className="text-slate-700 leading-relaxed bg-slate-50 rounded-xl p-4">
                      {scenario.context}
                    </p>
                  </div>

                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-slate-900 flex items-center">
                      <span className="w-5 h-5 text-purple-500 mr-2">💬</span>
                      Emma says:
                    </h3>
                    <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-4 border-l-4 border-purple-400">
                      <p className="text-slate-800 font-medium italic">
                        "{scenario.childStatement}"
                      </p>
                    </div>
                  </div>
                </div>

                <div className="mt-8 text-center">
                  <button
                    onClick={() => setCurrentStep('response')}
                    className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-8 py-4 rounded-2xl font-semibold text-lg hover:from-indigo-700 hover:to-purple-700 transition-all duration-300 shadow-xl hover:shadow-2xl transform hover:-translate-y-1"
                  >
                    How would you respond? →
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Response Input Step */}
        {currentStep === 'response' && (
          <div className="space-y-8">
            <div className="text-center">
              <div className="inline-flex items-center px-4 py-2 rounded-full bg-purple-100 text-purple-800 text-sm font-medium mb-4">
                <Brain className="w-4 h-4 mr-2" />
                Your Teaching Response
              </div>
              <h2 className="text-3xl font-bold text-slate-900 mb-2">What would you say to Emma?</h2>
              <p className="text-lg text-slate-600">Think about how to extend her learning while honoring her curiosity</p>
            </div>

            {/* Context Reminder */}
            <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-2xl p-6 border border-purple-200">
              <div className="flex items-start space-x-4">
                <div className="text-3xl">👧🏽</div>
                <div>
                  <h3 className="font-semibold text-purple-900">Emma just said:</h3>
                  <p className="text-purple-800 italic mt-1">"{scenario.childStatement}"</p>
                </div>
              </div>
            </div>

            {/* Response Input */}
            <div className="bg-white rounded-3xl shadow-xl p-8">
              <label htmlFor="response" className="block text-lg font-medium text-slate-900 mb-4">
                Your response as the educator:
              </label>
              <textarea
                id="response"
                value={userResponse}
                onChange={(e) => setUserResponse(e.target.value)}
                placeholder="Type what you would say to Emma here..."
                className="w-full h-40 px-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 resize-none text-lg"
              />

              <div className="flex justify-between items-center mt-6">
                <button
                  onClick={() => setCurrentStep('scene')}
                  className="text-slate-500 hover:text-slate-700 px-4 py-2 rounded-lg transition-colors flex items-center"
                >
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back to Scene
                </button>

                <button
                  onClick={handleSubmitResponse}
                  disabled={!userResponse.trim()}
                  className="bg-gradient-to-r from-emerald-600 to-teal-600 text-white px-8 py-3 rounded-xl font-semibold hover:from-emerald-700 hover:to-teal-700 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                >
                  <Send className="w-4 h-4 mr-2" />
                  Get AI Coaching
                </button>
              </div>
            </div>
          </div>
        )}

        {/* AI Feedback Step */}
        {currentStep === 'feedback' && (
          <div className="space-y-8">
            <div className="text-center">
              <div className="inline-flex items-center px-4 py-2 rounded-full bg-emerald-100 text-emerald-800 text-sm font-medium mb-4">
                <Brain className="w-4 h-4 mr-2" />
                AI Coaching Analysis
              </div>
              <h2 className="text-3xl font-bold text-slate-900 mb-2">Your Teaching Response Analysis</h2>
            </div>

            {/* User Response Display */}
            <div className="bg-slate-50 rounded-2xl p-6 border border-slate-200">
              <h3 className="font-semibold text-slate-900 mb-3">You said:</h3>
              <p className="text-slate-800 italic bg-white rounded-xl p-4 border border-slate-200">
                "{userResponse}"
              </p>
            </div>

            {/* Analysis Results */}
            {isAnalyzing ? (
              <div className="bg-white rounded-3xl shadow-xl p-12 text-center">
                <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-gradient-to-r from-indigo-600 to-purple-600 flex items-center justify-center animate-pulse">
                  <Brain className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-slate-900 mb-2">Analyzing your response...</h3>
                <p className="text-slate-600">Our AI is evaluating pedagogical quality and providing personalized coaching</p>
              </div>
            ) : feedback && (
              <div className="bg-white rounded-3xl shadow-xl overflow-hidden">
                {/* Score Header */}
                <div className={`p-6 ${feedback.qualityBg}`}>
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-3xl font-bold text-slate-900">{feedback.score}/100</div>
                      <div className={`text-lg font-semibold capitalize ${feedback.qualityColor}`}>
                        {feedback.quality} Response
                      </div>
                    </div>
                    <div className="w-20 h-20 rounded-full bg-white/80 flex items-center justify-center">
                      <div className="text-3xl">
                        {feedback.quality === 'exemplary' ? '🌟' : feedback.quality === 'proficient' ? '👍' : '📈'}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="p-8 space-y-8">
                  {/* Strengths */}
                  {feedback.strengths.length > 0 && (
                    <div>
                      <h3 className="text-lg font-semibold text-emerald-900 mb-4 flex items-center">
                        <span className="w-5 h-5 text-emerald-600 mr-2">✅</span>
                        Strengths in Your Response
                      </h3>
                      <div className="space-y-3">
                        {feedback.strengths.map((strength, index) => (
                          <div key={index} className="bg-emerald-50 rounded-xl p-4 border border-emerald-200">
                            <p className="text-emerald-800">{strength}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Areas for Growth */}
                  {feedback.improvements.length > 0 && (
                    <div>
                      <h3 className="text-lg font-semibold text-blue-900 mb-4 flex items-center">
                        <span className="w-5 h-5 text-blue-600 mr-2">💡</span>
                        Opportunities for Growth
                      </h3>
                      <div className="space-y-3">
                        {feedback.improvements.map((improvement, index) => (
                          <div key={index} className="bg-blue-50 rounded-xl p-4 border border-blue-200">
                            <p className="text-blue-800">{improvement}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Research Note */}
                  <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
                    <p className="text-sm text-slate-600">
                      <strong>Research Foundation:</strong> {feedback.researchNote}
                    </p>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex flex-col sm:flex-row gap-4 pt-6">
                    <button
                      onClick={restartDemo}
                      className="flex-1 bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-6 py-3 rounded-xl font-semibold hover:from-indigo-700 hover:to-purple-700 transition-all duration-300 shadow-lg hover:shadow-xl"
                    >
                      Try Another Scenario
                    </button>
                    <button
                      onClick={onBackToHome}
                      className="flex-1 bg-white border border-slate-300 text-slate-700 px-6 py-3 rounded-xl font-semibold hover:bg-slate-50 transition-colors shadow-lg"
                    >
                      Back to Home
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default InteractiveCoachingDemo;