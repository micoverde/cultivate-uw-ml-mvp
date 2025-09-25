import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadialBarChart,
  RadialBar,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  Area,
  AreaChart
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Target,
  Award,
  Brain,
  Users,
  MessageSquare,
  Clock,
  Zap,
  ChevronRight,
  Info,
  CheckCircle,
  AlertCircle,
  XCircle
} from 'lucide-react';

const ProfessionalVisualization = ({ results, scenarioContext }) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [hoveredMetric, setHoveredMetric] = useState(null);

  // Color palette following Microsoft Fluent Design
  const colorPalette = {
    primary: '#2E8B57',      // Cultivate Green
    secondary: '#4169E1',     // Royal Blue
    accent: '#FF6B6B',        // Coral
    success: '#10B981',       // Emerald
    warning: '#F59E0B',       // Amber
    danger: '#EF4444',        // Red
    neutral: '#6B7280',       // Gray
    surface: '#F8FAFC',       // Light surface
    text: '#1F2937'           // Dark text
  };

  // Process ML results for visualization
  const mlMetrics = useMemo(() => {
    if (!results?.ml_predictions) return [];

    const { ml_predictions } = results;
    return [
      {
        name: 'Question Quality',
        value: Math.round(ml_predictions.question_quality * 100),
        score: ml_predictions.question_quality,
        icon: MessageSquare,
        description: 'Quality of open-ended questions that promote thinking',
        benchmark: 70,
        trend: 'up'
      },
      {
        name: 'Wait Time',
        value: Math.round(ml_predictions.wait_time_appropriate * 100),
        score: ml_predictions.wait_time_appropriate,
        icon: Clock,
        description: 'Appropriate pause time for child processing and response',
        benchmark: 65,
        trend: 'stable'
      },
      {
        name: 'Scaffolding',
        value: Math.round(ml_predictions.scaffolding_present * 100),
        score: ml_predictions.scaffolding_present,
        icon: TrendingUp,
        description: 'Support provided to help children reach the next level',
        benchmark: 75,
        trend: 'up'
      },
      {
        name: 'Open Questions',
        value: Math.round(ml_predictions.open_ended_questions * 100),
        score: ml_predictions.open_ended_questions,
        icon: Brain,
        description: 'Use of questions that encourage deeper thinking',
        benchmark: 60,
        trend: 'down'
      }
    ];
  }, [results]);

  // Process CLASS scores for radar visualization
  const classData = useMemo(() => {
    if (!results?.class_scores) return [];

    return [
      { metric: 'Emotional Support', score: results.class_scores.emotional_support * 10, fullMark: 10 },
      { metric: 'Classroom Organization', score: results.class_scores.classroom_organization * 10, fullMark: 10 },
      { metric: 'Instructional Support', score: results.class_scores.instructional_support * 10, fullMark: 10 }
    ];
  }, [results]);

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.6,
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: {
      opacity: 1,
      x: 0,
      transition: { duration: 0.4 }
    }
  };

  const getScoreColor = (score) => {
    if (score >= 0.7) return colorPalette.success;
    if (score >= 0.5) return colorPalette.warning;
    return colorPalette.danger;
  };

  const getScoreIcon = (score) => {
    if (score >= 0.7) return CheckCircle;
    if (score >= 0.5) return AlertCircle;
    return XCircle;
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Target },
    { id: 'detailed', label: 'Detailed Analysis', icon: BarChart },
    { id: 'trends', label: 'Quality Trends', icon: TrendingUp },
    { id: 'insights', label: 'AI Insights', icon: Brain }
  ];

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-4 rounded-xl shadow-lg border border-gray-200/50">
          <p className="font-semibold text-gray-900">{label}</p>
          <p className="text-sm text-gray-600 mt-1">
            Score: <span className="font-medium">{payload[0].value}%</span>
          </p>
          {payload[0].payload.description && (
            <p className="text-xs text-gray-500 mt-2 max-w-48">
              {payload[0].payload.description}
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="w-full max-w-7xl mx-auto p-6 space-y-8"
    >
      {/* Header Section */}
      <motion.div variants={itemVariants} className="bg-white rounded-2xl shadow-lg p-8 border border-gray-100">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
          <div className="flex items-center space-x-4">
            <div className="w-16 h-16 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg">
              <Award className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Analysis Results</h1>
              {scenarioContext && (
                <p className="text-lg text-gray-600 mt-1">{scenarioContext.title}</p>
              )}
              <div className="flex items-center space-x-4 mt-2">
                <span className="text-sm text-gray-500">
                  Processing time: {results?.processing_time?.toFixed(2)}s
                </span>
                <span className="text-sm text-gray-500">•</span>
                <span className="text-sm text-gray-500">
                  {new Date(results?.completed_at).toLocaleString()}
                </span>
              </div>
            </div>
          </div>

          {/* Quick Score Overview */}
          <div className="flex items-center space-x-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-emerald-600">
                {Math.round((mlMetrics.reduce((sum, m) => sum + m.score, 0) / mlMetrics.length) * 100)}%
              </div>
              <div className="text-sm text-gray-600 font-medium">Overall Quality</div>
            </div>
            <div className="w-16 h-16">
              <ResponsiveContainer width="100%" height="100%">
                <RadialBarChart cx="50%" cy="50%" innerRadius="60%" outerRadius="90%"
                               data={[{ value: Math.round((mlMetrics.reduce((sum, m) => sum + m.score, 0) / mlMetrics.length) * 100) }]}>
                  <RadialBar dataKey="value" cornerRadius={10} fill={colorPalette.primary} />
                </RadialBarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Navigation Tabs */}
      <motion.div variants={itemVariants} className="bg-white rounded-xl shadow-sm border border-gray-100">
        <div className="flex flex-wrap border-b border-gray-200">
          {tabs.map((tab) => {
            const IconComponent = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-6 py-4 font-medium text-sm transition-colors relative ${
                  activeTab === tab.id
                    ? 'text-indigo-600 border-b-2 border-indigo-600'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <IconComponent className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </motion.div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        {activeTab === 'overview' && (
          <motion.div
            key="overview"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-8"
          >
            {/* ML Metrics Cards */}
            <div className="space-y-6">
              <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                <Zap className="w-5 h-5 mr-2 text-indigo-600" />
                ML Analysis Metrics
              </h2>

              {mlMetrics.map((metric, index) => {
                const IconComponent = metric.icon;
                const ScoreIcon = getScoreIcon(metric.score);

                return (
                  <motion.div
                    key={metric.name}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 hover:shadow-md transition-shadow"
                    onMouseEnter={() => setHoveredMetric(metric.name)}
                    onMouseLeave={() => setHoveredMetric(null)}
                  >
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center space-x-3">
                        <div className={`w-10 h-10 rounded-lg flex items-center justify-center`}
                             style={{ backgroundColor: `${getScoreColor(metric.score)}20` }}>
                          <IconComponent className="w-5 h-5" style={{ color: getScoreColor(metric.score) }} />
                        </div>
                        <div>
                          <h3 className="font-semibold text-gray-900">{metric.name}</h3>
                          <p className="text-sm text-gray-600 max-w-xs">{metric.description}</p>
                        </div>
                      </div>

                      <div className="text-right flex items-center space-x-2">
                        <ScoreIcon className="w-5 h-5" style={{ color: getScoreColor(metric.score) }} />
                        <span className="text-2xl font-bold" style={{ color: getScoreColor(metric.score) }}>
                          {metric.value}%
                        </span>
                      </div>
                    </div>

                    {/* Progress bar */}
                    <div className="relative">
                      <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                        <motion.div
                          className="h-2 rounded-full"
                          style={{ backgroundColor: getScoreColor(metric.score) }}
                          initial={{ width: 0 }}
                          animate={{ width: `${metric.value}%` }}
                          transition={{ delay: index * 0.1 + 0.3, duration: 0.8 }}
                        />
                      </div>

                      {/* Benchmark line */}
                      <div
                        className="absolute top-0 h-2 w-0.5 bg-gray-400"
                        style={{ left: `${metric.benchmark}%` }}
                      >
                        <div className="absolute -top-6 -left-6 text-xs text-gray-500 font-medium">
                          Target: {metric.benchmark}%
                        </div>
                      </div>
                    </div>

                    {/* Performance indicator */}
                    <div className="flex items-center justify-between mt-3">
                      <div className="flex items-center space-x-1">
                        {metric.value >= metric.benchmark ? (
                          <TrendingUp className="w-4 h-4 text-emerald-500" />
                        ) : (
                          <TrendingDown className="w-4 h-4 text-red-500" />
                        )}
                        <span className={`text-sm font-medium ${
                          metric.value >= metric.benchmark ? 'text-emerald-600' : 'text-red-600'
                        }`}>
                          {metric.value >= metric.benchmark ? 'Above Target' : 'Below Target'}
                        </span>
                      </div>

                      <div className="text-sm text-gray-500">
                        {Math.abs(metric.value - metric.benchmark)}% {metric.value >= metric.benchmark ? 'above' : 'below'}
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>

            {/* CLASS Framework Radar */}
            <div className="space-y-6">
              <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                <Target className="w-5 h-5 mr-2 text-indigo-600" />
                CLASS Framework Scores
              </h2>

              <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
                <div style={{ width: '100%', height: '300px' }}>
                  <ResponsiveContainer>
                    <RadialBarChart cx="50%" cy="50%" innerRadius="20%" outerRadius="80%" data={classData}>
                      <RadialBar
                        dataKey="score"
                        cornerRadius={10}
                        fill={colorPalette.primary}
                        label={{ position: 'insideStart', fill: '#fff', fontSize: 12 }}
                      />
                      <Tooltip content={<CustomTooltip />} />
                    </RadialBarChart>
                  </ResponsiveContainer>
                </div>

                {/* CLASS Score Details */}
                <div className="grid grid-cols-1 gap-4 mt-6">
                  {classData.map((item, index) => (
                    <div key={item.metric} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <span className="font-medium text-gray-900">{item.metric}</span>
                      <div className="flex items-center space-x-2">
                        <span className="text-lg font-bold text-indigo-600">
                          {item.score.toFixed(1)}
                        </span>
                        <span className="text-sm text-gray-500">/ 10</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {activeTab === 'detailed' && (
          <motion.div
            key="detailed"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="bg-white rounded-xl shadow-sm border border-gray-100 p-8"
          >
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Detailed Analysis Breakdown</h2>

            <div style={{ width: '100%', height: '400px' }}>
              <ResponsiveContainer>
                <BarChart data={mlMetrics} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 12 }}
                    stroke="#6B7280"
                  />
                  <YAxis
                    tick={{ fontSize: 12 }}
                    stroke="#6B7280"
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar
                    dataKey="value"
                    fill={colorPalette.primary}
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default ProfessionalVisualization;