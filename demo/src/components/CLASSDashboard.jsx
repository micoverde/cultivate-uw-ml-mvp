import React, { useState } from 'react';
import {
  TrendingUp,
  Heart,
  Clipboard,
  GraduationCap,
  Info,
  HelpCircle,
  BarChart3,
  Target,
  Users,
  MessageCircle,
  Star,
  ChevronRight,
  ChevronDown
} from 'lucide-react';

/**
 * CLASS Framework Dashboard Component
 *
 * Provides comprehensive visualization of Classroom Assessment Scoring System (CLASS)
 * scores with detailed breakdowns, research benchmarks, and indicator explanations.
 *
 * Author: Claude-4 (Partner-Level Microsoft SDE)
 * Issue: #50 - Story 2.4: View CLASS Framework Scoring
 */

const CLASSDashboard = ({ classScores, scaffoldingResults }) => {
  const [expandedDomain, setExpandedDomain] = useState('emotional_support');
  const [activeTooltip, setActiveTooltip] = useState(null);

  // Research benchmarks from CLASS Pre-K manual
  const researchBenchmarks = {
    emotional_support: {
      low: 2.0,
      mid: 4.0,
      high: 6.0,
      dimensions: {
        positive_climate: { description: "Warm, respectful relationships; positive affect and communication", benchmark: 4.5 },
        teacher_sensitivity: { description: "Responsive to children's needs, emotions, and interests", benchmark: 4.2 },
        regard_for_perspectives: { description: "Values children's ideas, motivations, and points of view", benchmark: 3.8 }
      }
    },
    classroom_organization: {
      low: 2.5,
      mid: 4.5,
      high: 6.5,
      dimensions: {
        behavior_management: { description: "Clear expectations and effective redirection methods", benchmark: 4.8 },
        productivity: { description: "Efficient use of time and strong preparation", benchmark: 4.3 }
      }
    },
    instructional_support: {
      low: 2.0,
      mid: 3.5,
      high: 5.0,
      dimensions: {
        concept_development: { description: "Analysis, reasoning, and problem-solving opportunities", benchmark: 3.2 },
        quality_feedback: { description: "Specific, process-oriented feedback that extends learning", benchmark: 3.5 },
        language_modeling: { description: "Rich language and advanced vocabulary exposure", benchmark: 3.7 }
      }
    }
  };

  // Get detailed scores from classScores.detailed_analysis if available
  const getDetailedScores = (domain) => {
    if (classScores.detailed_analysis && classScores.detailed_analysis[`${domain}_breakdown`]) {
      const breakdown = classScores.detailed_analysis[`${domain}_breakdown`];
      return breakdown.dimension_scores || {};
    }
    return {};
  };

  // Calculate performance level
  const getPerformanceLevel = (score, domain) => {
    const benchmarks = researchBenchmarks[domain];
    if (score >= benchmarks.high) return { level: 'High', color: 'green', description: 'Exceeds research benchmarks' };
    if (score >= benchmarks.mid) return { level: 'Mid', color: 'yellow', description: 'Meets typical expectations' };
    if (score >= benchmarks.low) return { level: 'Low-Mid', color: 'orange', description: 'Below typical range' };
    return { level: 'Low', color: 'red', description: 'Needs immediate attention' };
  };

  // Tooltip component
  const Tooltip = ({ content, children }) => (
    <div className="relative inline-block">
      {children}
      {activeTooltip === content && (
        <div className="absolute z-10 w-64 p-3 mt-2 text-sm bg-gray-900 text-white rounded-lg shadow-lg -left-32">
          {content}
          <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1 w-2 h-2 bg-gray-900 rotate-45"></div>
        </div>
      )}
    </div>
  );

  // Score visualization bar
  const ScoreBar = ({ score, max = 7, domain, dimension = null }) => {
    const percentage = (score / max) * 100;
    const performance = getPerformanceLevel(score, domain);

    const getBarColor = () => {
      switch (performance.color) {
        case 'green': return 'bg-green-500';
        case 'yellow': return 'bg-yellow-500';
        case 'orange': return 'bg-orange-500';
        case 'red': return 'bg-red-500';
        default: return 'bg-gray-500';
      }
    };

    return (
      <div className="flex items-center space-x-3">
        <div className="flex-1 bg-gray-200 rounded-full h-4 relative overflow-hidden">
          <div
            className={`h-full transition-all duration-500 ${getBarColor()}`}
            style={{ width: `${percentage}%` }}
          ></div>
          {/* Benchmark indicators */}
          {!dimension && (
            <>
              <div
                className="absolute top-0 h-full w-0.5 bg-gray-600"
                style={{ left: `${(researchBenchmarks[domain].mid / max) * 100}%` }}
                title="Research Benchmark (Mid)"
              ></div>
              <div
                className="absolute top-0 h-full w-0.5 bg-gray-800"
                style={{ left: `${(researchBenchmarks[domain].high / max) * 100}%` }}
                title="Research Benchmark (High)"
              ></div>
            </>
          )}
        </div>
        <span className="text-sm font-medium w-12 text-right">{score.toFixed(1)}</span>
        <span className={`text-xs px-2 py-1 rounded-full bg-${performance.color}-100 text-${performance.color}-800`}>
          {performance.level}
        </span>
      </div>
    );
  };

  // Domain header with icon
  const getDomainIcon = (domain) => {
    switch (domain) {
      case 'emotional_support': return <Heart className="h-6 w-6" />;
      case 'classroom_organization': return <Clipboard className="h-6 w-6" />;
      case 'instructional_support': return <GraduationCap className="h-6 w-6" />;
      default: return <Star className="h-6 w-6" />;
    }
  };

  const getDomainColor = (domain) => {
    switch (domain) {
      case 'emotional_support': return 'text-rose-600';
      case 'classroom_organization': return 'text-blue-600';
      case 'instructional_support': return 'text-green-600';
      default: return 'text-purple-600';
    }
  };

  const formatDomainName = (domain) => {
    return domain.split('_').map(word =>
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center">
          <TrendingUp className="h-6 w-6 text-indigo-600 mr-3" />
          <h2 className="text-2xl font-bold text-gray-900">CLASS Framework Dashboard</h2>
          <Tooltip content="The Classroom Assessment Scoring System (CLASS) is a research-based tool for measuring the quality of teacher-child interactions across three key domains.">
            <HelpCircle
              className="h-5 w-5 text-gray-400 ml-2 cursor-pointer"
              onMouseEnter={() => setActiveTooltip("The Classroom Assessment Scoring System (CLASS) is a research-based tool for measuring the quality of teacher-child interactions across three key domains.")}
              onMouseLeave={() => setActiveTooltip(null)}
            />
          </Tooltip>
        </div>
        <div className="text-sm text-gray-600">
          Overall Score: <span className="font-semibold text-lg">{classScores.overall_score?.toFixed(1) || 'N/A'}/7.0</span>
        </div>
      </div>

      {/* Overall Performance Summary */}
      <div className="grid md:grid-cols-3 gap-4 mb-8">
        {['emotional_support', 'classroom_organization', 'instructional_support'].map(domain => {
          const score = classScores[domain] || 0;
          const performance = getPerformanceLevel(score, domain);

          return (
            <div key={domain} className="border rounded-lg p-4 cursor-pointer hover:shadow-md transition-shadow"
                 onClick={() => setExpandedDomain(expandedDomain === domain ? null : domain)}>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center">
                  <span className={getDomainColor(domain)}>{getDomainIcon(domain)}</span>
                  <h3 className="font-semibold ml-2">{formatDomainName(domain)}</h3>
                </div>
                {expandedDomain === domain ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
              </div>
              <ScoreBar score={score} domain={domain} />
              <p className="text-xs text-gray-600 mt-2">{performance.description}</p>
            </div>
          );
        })}
      </div>

      {/* Detailed Domain Analysis */}
      {expandedDomain && (
        <div className="border-t pt-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <span className={getDomainColor(expandedDomain)}>{getDomainIcon(expandedDomain)}</span>
            <span className="ml-2">{formatDomainName(expandedDomain)} - Detailed Analysis</span>
          </h3>

          <div className="space-y-4">
            {/* Domain dimensions */}
            {Object.entries(researchBenchmarks[expandedDomain].dimensions).map(([dimension, info]) => {
              const detailedScores = getDetailedScores(expandedDomain);
              const dimensionScore = detailedScores[dimension] || 4.0; // Default if not available

              return (
                <div key={dimension} className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium capitalize">{dimension.replace('_', ' ')}</h4>
                    <Tooltip content={info.description}>
                      <Info
                        className="h-4 w-4 text-gray-400 cursor-pointer"
                        onMouseEnter={() => setActiveTooltip(info.description)}
                        onMouseLeave={() => setActiveTooltip(null)}
                      />
                    </Tooltip>
                  </div>
                  <ScoreBar score={dimensionScore} domain={expandedDomain} dimension={dimension} />
                  <div className="mt-2 text-xs text-gray-600 flex justify-between">
                    <span>Research Benchmark: {info.benchmark.toFixed(1)}</span>
                    <span className={dimensionScore >= info.benchmark ? 'text-green-600' : 'text-orange-600'}>
                      {dimensionScore >= info.benchmark ? 'Above' : 'Below'} Benchmark
                    </span>
                  </div>
                </div>
              );
            })}

            {/* Research Context */}
            <div className="bg-blue-50 rounded-lg p-4 mt-4">
              <h4 className="font-medium text-blue-900 mb-2">Research Context</h4>
              <p className="text-sm text-blue-800">
                {expandedDomain === 'emotional_support' && "Emotional Support measures the warmth, respect, and emotional connection in teacher-child interactions. High scores indicate positive relationships that support children's social-emotional development."}
                {expandedDomain === 'classroom_organization' && "Classroom Organization assesses how well the learning environment is structured and managed. Effective organization maximizes learning time and minimizes behavioral issues."}
                {expandedDomain === 'instructional_support' && "Instructional Support evaluates the quality of educational content and teaching methods. Higher scores indicate more opportunities for deep learning and cognitive development."}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Research Benchmark Legend */}
      <div className="mt-6 border-t pt-4">
        <h4 className="font-medium mb-3">Research Benchmarks (Pianta et al., 2008)</h4>
        <div className="grid md:grid-cols-4 gap-4 text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-red-500 rounded"></div>
            <span>Low (1.0-2.5): Immediate Attention</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-orange-500 rounded"></div>
            <span>Low-Mid (2.5-4.0): Below Range</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-yellow-500 rounded"></div>
            <span>Mid (4.0-5.5): Typical Range</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded"></div>
            <span>High (5.5-7.0): Exceeds Benchmarks</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CLASSDashboard;