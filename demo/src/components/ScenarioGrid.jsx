import React, { useState, useMemo } from 'react';
import { Search, Filter, Users, BookOpen, Target, Star } from 'lucide-react';
import ScenarioCard from './ScenarioCard';
import { educatorScenarios, getAgeGroupLabel, getInteractionTypeLabel } from '../data/scenarios';

/**
 * ScenarioGrid Component
 *
 * Professional scenario selection interface with filtering capabilities.
 * Designed for stakeholder presentations and funding demos.
 *
 * Feature 3: Professional Demo Interface
 * Story 3.1: Select realistic educator scenarios
 *
 * Author: Claude (Partner-Level Microsoft SDE)
 * Issue: #48 - Scenario Selection Interface
 */

const ScenarioGrid = ({ onScenarioSelect }) => {
  // Filter state
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilters, setActiveFilters] = useState({
    ageGroups: [],
    qualityLevels: [],
    interactionTypes: []
  });
  const [sortBy, setSortBy] = useState('title');

  // Get unique filter options from scenarios
  const filterOptions = useMemo(() => {
    const ageGroups = [...new Set(educatorScenarios.map(s => s.ageGroup))];
    const qualityLevels = [...new Set(educatorScenarios.map(s => s.expectedQuality))];
    const interactionTypes = [...new Set(educatorScenarios.map(s => s.interactionType))];

    return {
      ageGroups: ageGroups.sort(),
      qualityLevels: ['exemplary', 'proficient', 'developing', 'struggling'].filter(q => qualityLevels.includes(q)),
      interactionTypes: interactionTypes.sort()
    };
  }, []);

  // Filter and sort scenarios
  const filteredScenarios = useMemo(() => {
    let filtered = educatorScenarios;

    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(scenario =>
        scenario.title.toLowerCase().includes(query) ||
        scenario.description.toLowerCase().includes(query) ||
        scenario.tags.some(tag => tag.toLowerCase().includes(query))
      );
    }

    // Apply category filters
    if (activeFilters.ageGroups.length > 0) {
      filtered = filtered.filter(scenario => activeFilters.ageGroups.includes(scenario.ageGroup));
    }

    if (activeFilters.qualityLevels.length > 0) {
      filtered = filtered.filter(scenario => activeFilters.qualityLevels.includes(scenario.expectedQuality));
    }

    if (activeFilters.interactionTypes.length > 0) {
      filtered = filtered.filter(scenario => activeFilters.interactionTypes.includes(scenario.interactionType));
    }

    // Apply sorting
    const sorted = [...filtered].sort((a, b) => {
      switch (sortBy) {
        case 'title':
          return a.title.localeCompare(b.title);
        case 'quality':
          const qualityOrder = { exemplary: 4, proficient: 3, developing: 2, struggling: 1 };
          return qualityOrder[b.expectedQuality] - qualityOrder[a.expectedQuality];
        case 'age':
          return a.ageGroup.localeCompare(b.ageGroup);
        case 'duration':
          return a.duration - b.duration;
        default:
          return 0;
      }
    });

    return sorted;
  }, [searchQuery, activeFilters, sortBy]);

  // Toggle filter functions
  const toggleFilter = (category, value) => {
    setActiveFilters(prev => ({
      ...prev,
      [category]: prev[category].includes(value)
        ? prev[category].filter(item => item !== value)
        : [...prev[category], value]
    }));
  };

  // Clear all filters
  const clearFilters = () => {
    setSearchQuery('');
    setActiveFilters({
      ageGroups: [],
      qualityLevels: [],
      interactionTypes: []
    });
    setSortBy('title');
  };

  // Get active filter count
  const activeFilterCount = activeFilters.ageGroups.length +
                           activeFilters.qualityLevels.length +
                           activeFilters.interactionTypes.length;

  return (
    <div className="scenario-grid-container">

      {/* Header Section */}
      <div className="mb-8">
        <div className="text-center mb-6">
          <h2 className="text-3xl font-bold text-gray-900 mb-3">
            Select an Educator Scenario
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Choose from realistic educator-child interactions to see our ML analysis in action.
            Each scenario demonstrates different aspects of educational quality and teaching practices.
          </p>
        </div>

        {/* Statistics Bar */}
        <div className="flex justify-center space-x-8 mb-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-indigo-600">{educatorScenarios.length}</div>
            <div className="text-sm text-gray-600">Scenarios</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">4</div>
            <div className="text-sm text-gray-600">Quality Levels</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">6</div>
            <div className="text-sm text-gray-600">Age Groups</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {filterOptions.interactionTypes.length}
            </div>
            <div className="text-sm text-gray-600">Interaction Types</div>
          </div>
        </div>
      </div>

      {/* Search and Filter Controls */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">

        {/* Search Bar */}
        <div className="mb-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search scenarios by title, description, or tags..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />
          </div>
        </div>

        {/* Filter and Sort Controls */}
        <div className="flex flex-wrap gap-4 items-center justify-between">

          {/* Filter Buttons */}
          <div className="flex flex-wrap gap-2">

            {/* Age Group Filter */}
            <div className="relative group">
              <button className="flex items-center space-x-2 px-3 py-2 bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 text-sm font-medium transition-colors">
                <Users className="h-4 w-4" />
                <span>Age Groups</span>
                {activeFilters.ageGroups.length > 0 && (
                  <span className="bg-indigo-600 text-white rounded-full px-2 py-0.5 text-xs">
                    {activeFilters.ageGroups.length}
                  </span>
                )}
              </button>

              {/* Age Group Dropdown */}
              <div className="absolute top-full left-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-10 min-w-48 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all">
                <div className="p-2">
                  {filterOptions.ageGroups.map(ageGroup => (
                    <label key={ageGroup} className="flex items-center space-x-2 p-2 hover:bg-gray-50 rounded cursor-pointer">
                      <input
                        type="checkbox"
                        checked={activeFilters.ageGroups.includes(ageGroup)}
                        onChange={() => toggleFilter('ageGroups', ageGroup)}
                        className="text-indigo-600 focus:ring-indigo-500"
                      />
                      <span className="text-sm">{getAgeGroupLabel(ageGroup)}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>

            {/* Quality Level Filter */}
            <div className="relative group">
              <button className="flex items-center space-x-2 px-3 py-2 bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 text-sm font-medium transition-colors">
                <Star className="h-4 w-4" />
                <span>Quality</span>
                {activeFilters.qualityLevels.length > 0 && (
                  <span className="bg-indigo-600 text-white rounded-full px-2 py-0.5 text-xs">
                    {activeFilters.qualityLevels.length}
                  </span>
                )}
              </button>

              <div className="absolute top-full left-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-10 min-w-48 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all">
                <div className="p-2">
                  {filterOptions.qualityLevels.map(quality => (
                    <label key={quality} className="flex items-center space-x-2 p-2 hover:bg-gray-50 rounded cursor-pointer">
                      <input
                        type="checkbox"
                        checked={activeFilters.qualityLevels.includes(quality)}
                        onChange={() => toggleFilter('qualityLevels', quality)}
                        className="text-indigo-600 focus:ring-indigo-500"
                      />
                      <span className="text-sm capitalize">{quality}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>

            {/* Interaction Type Filter */}
            <div className="relative group">
              <button className="flex items-center space-x-2 px-3 py-2 bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 text-sm font-medium transition-colors">
                <BookOpen className="h-4 w-4" />
                <span>Type</span>
                {activeFilters.interactionTypes.length > 0 && (
                  <span className="bg-indigo-600 text-white rounded-full px-2 py-0.5 text-xs">
                    {activeFilters.interactionTypes.length}
                  </span>
                )}
              </button>

              <div className="absolute top-full left-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-10 min-w-48 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all">
                <div className="p-2">
                  {filterOptions.interactionTypes.map(type => (
                    <label key={type} className="flex items-center space-x-2 p-2 hover:bg-gray-50 rounded cursor-pointer">
                      <input
                        type="checkbox"
                        checked={activeFilters.interactionTypes.includes(type)}
                        onChange={() => toggleFilter('interactionTypes', type)}
                        className="text-indigo-600 focus:ring-indigo-500"
                      />
                      <span className="text-sm">{getInteractionTypeLabel(type)}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Sort and Clear Controls */}
          <div className="flex items-center space-x-3">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500"
            >
              <option value="title">Sort by Title</option>
              <option value="quality">Sort by Quality</option>
              <option value="age">Sort by Age</option>
              <option value="duration">Sort by Duration</option>
            </select>

            {(activeFilterCount > 0 || searchQuery) && (
              <button
                onClick={clearFilters}
                className="px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
              >
                Clear All
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Results Summary */}
      <div className="mb-4">
        <p className="text-gray-600">
          Showing <span className="font-semibold">{filteredScenarios.length}</span> of{' '}
          <span className="font-semibold">{educatorScenarios.length}</span> scenarios
          {(activeFilterCount > 0 || searchQuery) && (
            <span className="ml-2 text-indigo-600">
              ({activeFilterCount} filter{activeFilterCount !== 1 ? 's' : ''} active)
            </span>
          )}
        </p>
      </div>

      {/* Scenario Grid */}
      {filteredScenarios.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {filteredScenarios.map((scenario) => (
            <ScenarioCard
              key={scenario.id}
              scenario={scenario}
              onSelect={onScenarioSelect}
              className="h-full"
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <div className="mb-4">
            <Filter className="h-12 w-12 text-gray-300 mx-auto" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No scenarios found</h3>
          <p className="text-gray-600 mb-4">
            Try adjusting your search terms or filters to find scenarios.
          </p>
          <button
            onClick={clearFilters}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
          >
            Clear All Filters
          </button>
        </div>
      )}
    </div>
  );
};

export default ScenarioGrid;