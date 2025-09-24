import React, { useState, useMemo } from 'react';
import {
  Brain,
  Target,
  Clock,
  TrendingUp,
  BookOpen,
  Users,
  ChevronDown,
  ChevronRight,
  CheckCircle,
  AlertCircle,
  Info
} from 'lucide-react';

/**
 * ScaffoldingVisualization Component
 *
 * Displays comprehensive scaffolding and ZPD analysis results with:
 * - Interactive transcript highlighting
 * - ZPD indicator identification
 * - Scaffolding technique visualization
 * - Research-backed recommendations
 *
 * Author: Claude-4 (Partner-Level Microsoft SDE)
 * Issue: #49 - Story 2.3: Scaffolding Technique Identification
 */

const ScaffoldingVisualization = ({ scaffoldingResults, originalTranscript }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [expandedSections, setExpandedSections] = useState({
    zpd_indicators: true,
    scaffolding_techniques: true,
    wait_time: false,
    fading_support: false
  });

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  // Process transcript for highlighting
  const highlightedTranscript = useMemo(() => {
    if (!scaffoldingResults || !originalTranscript) return originalTranscript;

    let highlighted = originalTranscript;
    const highlights = [];

    // Add ZPD indicator highlights
    Object.entries(scaffoldingResults.zpd_indicators || {}).forEach(([zpdType, data]) => {
      data.indicators?.forEach(indicator => {
        highlights.push({
          text: indicator.evidence,
          type: 'zpd',
          subtype: zpdType,
          confidence: indicator.confidence,
          research: indicator.research_backing
        });
      });
    });

    // Add scaffolding technique highlights
    Object.entries(scaffoldingResults.scaffolding_techniques || {}).forEach(([techType, data]) => {
      data.techniques_found?.forEach(technique => {
        highlights.push({
          text: technique.evidence,
          type: 'scaffolding',
          subtype: techType,
          effectiveness: technique.effectiveness_score
        });
      });
    });

    // Sort by length (longest first) to avoid nested highlighting issues
    highlights.sort((a, b) => b.text.length - a.text.length);

    // Apply highlights
    highlights.forEach((highlight, index) => {
      const className = getHighlightClassName(highlight.type, highlight.subtype);
      const regex = new RegExp(escapeRegExp(highlight.text), 'gi');
      highlighted = highlighted.replace(regex,
        `<mark class="${className}" data-highlight-id="${index}">${highlight.text}</mark>`
      );
    });

    return { __html: highlighted };
  }, [scaffoldingResults, originalTranscript]);

  const getHighlightClassName = (type, subtype) => {
    const baseClasses = 'px-1 py-0.5 rounded text-sm font-medium cursor-pointer transition-colors';

    if (type === 'zpd') {
      switch (subtype) {
        case 'appropriate_challenge': return `${baseClasses} bg-blue-100 text-blue-800 hover:bg-blue-200`;
        case 'guided_discovery': return `${baseClasses} bg-green-100 text-green-800 hover:bg-green-200`;
        case 'developmental_matching': return `${baseClasses} bg-purple-100 text-purple-800 hover:bg-purple-200`;
        default: return `${baseClasses} bg-gray-100 text-gray-800 hover:bg-gray-200`;
      }
    } else if (type === 'scaffolding') {
      switch (subtype) {
        case 'modeling_thinking': return `${baseClasses} bg-yellow-100 text-yellow-800 hover:bg-yellow-200`;
        case 'graduated_prompting': return `${baseClasses} bg-orange-100 text-orange-800 hover:bg-orange-200`;
        case 'collaborative_construction': return `${baseClasses} bg-pink-100 text-pink-800 hover:bg-pink-200`;
        case 'fading_support': return `${baseClasses} bg-indigo-100 text-indigo-800 hover:bg-indigo-200`;
        default: return `${baseClasses} bg-gray-100 text-gray-800 hover:bg-gray-200`;
      }
    }

    return `${baseClasses} bg-gray-100 text-gray-800 hover:bg-gray-200`;
  };

  const escapeRegExp = (string) => {
    return string.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
  };

  const renderOverallAssessment = () => {
    const assessment = scaffoldingResults?.overall_assessment;
    if (!assessment) return null;

    const score = assessment.overall_scaffolding_zpd_score || 0;
    const getScoreColor = (score) => {
      if (score >= 0.8) return 'text-green-600 bg-green-50';
      if (score >= 0.6) return 'text-yellow-600 bg-yellow-50';
      if (score >= 0.4) return 'text-orange-600 bg-orange-50';
      return 'text-red-600 bg-red-50';
    };

    return (
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
          <Target className="h-6 w-6 text-indigo-600 mr-2" />
          Overall Scaffolding & ZPD Assessment
        </h3>

        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <div className={`rounded-lg p-4 ${getScoreColor(score)}`}>
              <div className="text-2xl font-bold">{(score * 100).toFixed(0)}%</div>
              <div className="text-sm font-medium">Overall Implementation Score</div>
            </div>
            <p className="mt-3 text-gray-600">{assessment.assessment_summary}</p>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">ZPD Implementation</span>
              <span className="font-medium">{((assessment.zpd_implementation_score || 0) * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Scaffolding Techniques</span>
              <span className="font-medium">{((assessment.scaffolding_technique_score || 0) * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Wait Time Implementation</span>
              <span className="font-medium">{((assessment.wait_time_implementation_score || 0) * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Fading Support</span>
              <span className="font-medium">{((assessment.fading_support_score || 0) * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderZPDIndicators = () => {
    const zpd = scaffoldingResults?.zpd_indicators;
    if (!zpd) return null;

    return (
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <button
          onClick={() => toggleSection('zpd_indicators')}
          className="w-full flex items-center justify-between text-xl font-bold text-gray-900 mb-4 hover:text-indigo-600 transition-colors"
        >
          <span className="flex items-center">
            <Brain className="h-6 w-6 text-blue-600 mr-2" />
            Zone of Proximal Development Indicators
          </span>
          {expandedSections.zpd_indicators ? <ChevronDown className="h-5 w-5" /> : <ChevronRight className="h-5 w-5" />}
        </button>

        {expandedSections.zpd_indicators && (
          <div className="space-y-4">
            {Object.entries(zpd).map(([zpdType, data]) => (
              <div key={zpdType} className="border-l-4 border-blue-200 pl-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-gray-800 capitalize">
                    {zpdType.replace('_', ' ')}
                  </h4>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-600">
                      {data.frequency} found
                    </span>
                    <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                      {((data.average_confidence || 0) * 100).toFixed(0)}% confidence
                    </span>
                  </div>
                </div>
                <p className="text-sm text-gray-600 mb-2">{data.description}</p>
                {data.indicators && data.indicators.length > 0 && (
                  <div className="space-y-2">
                    {data.indicators.slice(0, 3).map((indicator, idx) => (
                      <div key={idx} className="bg-blue-50 p-2 rounded text-sm">
                        <div className="font-medium text-blue-900">"{indicator.evidence}"</div>
                        <div className="text-blue-700 text-xs mt-1">Line {indicator.line_number}</div>
                      </div>
                    ))}
                  </div>
                )}
                <div className="text-xs text-gray-500 mt-2">
                  Research: {data.research_backing}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  const renderScaffoldingTechniques = () => {
    const scaffolding = scaffoldingResults?.scaffolding_techniques;
    if (!scaffolding) return null;

    return (
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <button
          onClick={() => toggleSection('scaffolding_techniques')}
          className="w-full flex items-center justify-between text-xl font-bold text-gray-900 mb-4 hover:text-indigo-600 transition-colors"
        >
          <span className="flex items-center">
            <Users className="h-6 w-6 text-green-600 mr-2" />
            Scaffolding Techniques Identified
          </span>
          {expandedSections.scaffolding_techniques ? <ChevronDown className="h-5 w-5" /> : <ChevronRight className="h-5 w-5" />}
        </button>

        {expandedSections.scaffolding_techniques && (
          <div className="space-y-4">
            {Object.entries(scaffolding).map(([techniqueType, data]) => (
              <div key={techniqueType} className="border-l-4 border-green-200 pl-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold text-gray-800 capitalize">
                    {techniqueType.replace('_', ' ')}
                  </h4>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-600">
                      {data.frequency} instances
                    </span>
                    <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">
                      {((data.average_effectiveness || 0) * 100).toFixed(0)}% effective
                    </span>
                  </div>
                </div>
                <p className="text-sm text-gray-600 mb-2">{data.description}</p>
                {data.techniques_found && data.techniques_found.length > 0 && (
                  <div className="space-y-2">
                    {data.techniques_found.slice(0, 2).map((technique, idx) => (
                      <div key={idx} className="bg-green-50 p-2 rounded text-sm">
                        <div className="font-medium text-green-900">"{technique.evidence}"</div>
                        <div className="flex justify-between items-center mt-1">
                          <span className="text-green-700 text-xs">Line {technique.line_number}</span>
                          <span className="text-green-600 text-xs">
                            {(technique.effectiveness_score * 100).toFixed(0)}% effectiveness
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                <div className="text-xs text-gray-500 mt-2">
                  Research: {data.research_citation}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  const renderRecommendations = () => {
    const recommendations = scaffoldingResults?.recommendations;
    if (!recommendations || recommendations.length === 0) return null;

    return (
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
          <BookOpen className="h-6 w-6 text-indigo-600 mr-2" />
          Research-Based Recommendations
        </h3>
        <div className="space-y-3">
          {recommendations.map((recommendation, index) => (
            <div key={index} className="flex items-start space-x-3 p-3 bg-indigo-50 rounded-lg">
              <CheckCircle className="h-5 w-5 text-indigo-600 mt-0.5 flex-shrink-0" />
              <p className="text-gray-800">{recommendation}</p>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderTranscriptHighlights = () => {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
          <Info className="h-6 w-6 text-purple-600 mr-2" />
          Interactive Transcript Analysis
        </h3>

        {/* Legend */}
        <div className="mb-4 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-medium text-gray-800 mb-2">Highlighting Legend:</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
            <div className="flex items-center space-x-2">
              <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded">ZPD Challenge</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="px-2 py-1 bg-green-100 text-green-800 rounded">Guided Discovery</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded">Think-Aloud</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="px-2 py-1 bg-orange-100 text-orange-800 rounded">Graduated Prompts</span>
            </div>
          </div>
        </div>

        {/* Highlighted transcript */}
        <div
          className="p-4 bg-gray-50 rounded-lg font-mono text-sm leading-relaxed whitespace-pre-wrap"
          dangerouslySetInnerHTML={highlightedTranscript}
        />
      </div>
    );
  };

  if (!scaffoldingResults) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 text-center">
        <AlertCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-600">No scaffolding analysis results available.</p>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Tab Navigation */}
      <div className="mb-6 border-b border-gray-200">
        <nav className="flex space-x-8">
          <button
            onClick={() => setActiveTab('overview')}
            className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'overview'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Overview & Assessment
          </button>
          <button
            onClick={() => setActiveTab('transcript')}
            className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'transcript'
                ? 'border-indigo-500 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Highlighted Transcript
          </button>
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div>
          {renderOverallAssessment()}
          {renderZPDIndicators()}
          {renderScaffoldingTechniques()}
          {renderRecommendations()}
        </div>
      )}

      {activeTab === 'transcript' && (
        <div>
          {renderTranscriptHighlights()}
        </div>
      )}
    </div>
  );
};

export default ScaffoldingVisualization;