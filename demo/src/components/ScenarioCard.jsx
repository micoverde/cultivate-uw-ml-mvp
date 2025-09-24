import React from 'react';
import { Clock, Users, Star, BookOpen, Target, ChevronRight } from 'lucide-react';
import { getQualityColor, getAgeGroupLabel, getInteractionTypeLabel } from '../data/scenarios';

/**
 * ScenarioCard Component
 *
 * Professional presentation-ready card for educator scenario selection.
 * Optimized for stakeholder demos and funding presentations.
 *
 * Feature 3: Professional Demo Interface
 * Story 3.1: Select realistic educator scenarios
 *
 * Author: Claude (Partner-Level Microsoft SDE)
 * Issue: #48 - Scenario Selection Interface
 */

const ScenarioCard = ({ scenario, onSelect, className = '' }) => {
  const {
    title,
    description,
    ageGroup,
    interactionType,
    expectedQuality,
    duration,
    complexity,
    expectedInsights,
    focusAreas
  } = scenario;

  // Get quality-based styling
  const qualityColor = getQualityColor(expectedQuality);
  const qualityColorClass = {
    exemplary: 'border-green-200 bg-green-50',
    proficient: 'border-blue-200 bg-blue-50',
    developing: 'border-amber-200 bg-amber-50',
    struggling: 'border-red-200 bg-red-50'
  }[expectedQuality] || 'border-gray-200 bg-gray-50';

  // Format duration for display
  const formatDuration = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    if (minutes === 0) return `${remainingSeconds}s`;
    if (remainingSeconds === 0) return `${minutes}m`;
    return `${minutes}m ${remainingSeconds}s`;
  };

  // Get complexity icon
  const getComplexityIcon = (level) => {
    switch (level) {
      case 'simple': return '●';
      case 'moderate': return '●●';
      case 'complex': return '●●●';
      default: return '●';
    }
  };

  // Get interaction type icon
  const getTypeIcon = (type) => {
    const icons = {
      lesson: '📚',
      play: '🎭',
      reading: '📖',
      'problem-solving': '🧩',
      transition: '⚡',
      outdoor: '🌳'
    };
    return icons[type] || '📋';
  };

  return (
    <div
      className={`scenario-card group cursor-pointer transition-all duration-300 hover:shadow-lg hover:-translate-y-1 ${className}`}
      onClick={() => onSelect(scenario)}
    >
      {/* Card Container */}
      <div className={`
        relative bg-white rounded-xl border-2 shadow-md overflow-hidden h-full
        ${qualityColorClass}
        group-hover:shadow-xl group-hover:border-indigo-300
      `}>

        {/* Quality Badge */}
        <div className="absolute top-3 right-3 z-10">
          <div
            className="px-3 py-1 rounded-full text-xs font-semibold text-white shadow-sm"
            style={{ backgroundColor: qualityColor }}
          >
            {expectedQuality.charAt(0).toUpperCase() + expectedQuality.slice(1)}
          </div>
        </div>

        {/* Card Content */}
        <div className="p-6 h-full flex flex-col">

          {/* Header Section */}
          <div className="mb-4">
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center space-x-2">
                <span className="text-2xl">{getTypeIcon(interactionType)}</span>
                <div className="text-xs text-gray-600 font-medium">
                  {getInteractionTypeLabel(interactionType)}
                </div>
              </div>
            </div>

            <h3 className="text-lg font-bold text-gray-900 mb-2 leading-tight">
              {title}
            </h3>

            <p className="text-sm text-gray-700 leading-relaxed line-clamp-2">
              {description}
            </p>
          </div>

          {/* Key Information */}
          <div className="flex-1 space-y-3">

            {/* Age and Duration Row */}
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center space-x-1 text-gray-600">
                <Users className="h-3 w-3" />
                <span>{getAgeGroupLabel(ageGroup)}</span>
              </div>

              <div className="flex items-center space-x-1 text-gray-600">
                <Clock className="h-3 w-3" />
                <span>{formatDuration(duration)}</span>
              </div>
            </div>

            {/* Complexity and Focus Areas */}
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center space-x-1 text-gray-600">
                <Target className="h-3 w-3" />
                <span>{getComplexityIcon(complexity)} {complexity}</span>
              </div>

              <div className="text-gray-500">
                {focusAreas.length} focus areas
              </div>
            </div>

            {/* Expected Insights Preview */}
            <div className="bg-white/60 rounded-lg p-3 border border-gray-200">
              <div className="flex items-center space-x-1 mb-2">
                <Star className="h-3 w-3 text-indigo-600" />
                <span className="text-xs font-medium text-indigo-900">Key Insights</span>
              </div>

              <div className="space-y-1">
                {expectedInsights.slice(0, 2).map((insight, index) => (
                  <div key={index} className="text-xs text-gray-700 flex items-start">
                    <span className="text-indigo-400 mr-1">•</span>
                    <span className="line-clamp-1">{insight}</span>
                  </div>
                ))}
                {expectedInsights.length > 2 && (
                  <div className="text-xs text-indigo-600 font-medium">
                    +{expectedInsights.length - 2} more insights
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Action Button */}
          <div className="mt-4 pt-3 border-t border-gray-200">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-indigo-700">
                Analyze This Scenario
              </span>
              <ChevronRight className="h-4 w-4 text-indigo-600 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>
        </div>

        {/* Hover Overlay */}
        <div className="absolute inset-0 bg-indigo-600/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />
      </div>
    </div>
  );
};

export default ScenarioCard;