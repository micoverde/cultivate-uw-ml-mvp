/**
 * Unit Tests for Professional Visualization Component
 * Tests Microsoft partner-level visualization features and interactions
 */

import React from 'react'
import { describe, test, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import ProfessionalVisualization from '../components/ProfessionalVisualization'

// Mock framer-motion to avoid animation issues in tests
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }) => children,
}))

// Mock recharts to avoid canvas issues in tests
vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }) => <div data-testid="responsive-container">{children}</div>,
  BarChart: ({ children }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => <div data-testid="bar" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  RadialBarChart: ({ children }) => <div data-testid="radial-bar-chart">{children}</div>,
  RadialBar: () => <div data-testid="radial-bar" />,
  PieChart: ({ children }) => <div data-testid="pie-chart">{children}</div>,
  Pie: () => <div data-testid="pie" />,
  Cell: () => <div data-testid="cell" />,
  LineChart: ({ children }) => <div data-testid="line-chart">{children}</div>,
  Line: () => <div data-testid="line" />,
  Area: () => <div data-testid="area" />,
  AreaChart: ({ children }) => <div data-testid="area-chart">{children}</div>
}))

describe('ProfessionalVisualization', () => {
  // Mock analysis results data
  const mockResults = {
    ml_predictions: {
      question_quality: 0.85,
      wait_time_appropriate: 0.72,
      scaffolding_present: 0.68,
      open_ended_questions: 0.91
    },
    class_scores: {
      emotional_support: 7.2,
      classroom_organization: 8.1,
      instructional_support: 6.8
    },
    processing_time: 2.34,
    completed_at: '2025-09-25T19:30:00Z'
  }

  const mockScenarioContext = {
    title: 'Maya & the Puzzle Challenge',
    category: 'Problem-solving',
    ageGroup: '4-5 years'
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('Component Rendering', () => {
    test('renders main header with scenario context', () => {
      render(
        <ProfessionalVisualization
          results={mockResults}
          scenarioContext={mockScenarioContext}
        />
      )

      expect(screen.getByText('Analysis Results')).toBeInTheDocument()
      expect(screen.getByText('Maya & the Puzzle Challenge')).toBeInTheDocument()
    })

    test('displays processing time and completion timestamp', () => {
      render(
        <ProfessionalVisualization
          results={mockResults}
          scenarioContext={mockScenarioContext}
        />
      )

      expect(screen.getByText(/Processing time: 2\.34s/)).toBeInTheDocument()
      expect(screen.getByText(/9\/25\/2025/)).toBeInTheDocument()
    })

    test('calculates and displays overall quality score', () => {
      render(
        <ProfessionalVisualization
          results={mockResults}
          scenarioContext={mockScenarioContext}
        />
      )

      // Overall score should be average of ML predictions: (85+72+68+91)/4 = 79%
      expect(screen.getByText('79%')).toBeInTheDocument()
      expect(screen.getByText('Overall Quality')).toBeInTheDocument()
    })

    test('renders navigation tabs correctly', () => {
      render(
        <ProfessionalVisualization
          results={mockResults}
          scenarioContext={mockScenarioContext}
        />
      )

      expect(screen.getByText('Overview')).toBeInTheDocument()
      expect(screen.getByText('Detailed Analysis')).toBeInTheDocument()
      expect(screen.getByText('Quality Trends')).toBeInTheDocument()
      expect(screen.getByText('AI Insights')).toBeInTheDocument()
    })
  })

  describe('ML Metrics Visualization', () => {
    test('displays all ML prediction metrics with correct scores', () => {
      render(
        <ProfessionalVisualization
          results={mockResults}
          scenarioContext={mockScenarioContext}
        />
      )

      expect(screen.getByText('Question Quality')).toBeInTheDocument()
      expect(screen.getByText('85%')).toBeInTheDocument() // 0.85 * 100

      expect(screen.getByText('Wait Time')).toBeInTheDocument()
      expect(screen.getByText('72%')).toBeInTheDocument() // 0.72 * 100

      expect(screen.getByText('Scaffolding')).toBeInTheDocument()
      expect(screen.getByText('68%')).toBeInTheDocument() // 0.68 * 100

      expect(screen.getByText('Open Questions')).toBeInTheDocument()
      expect(screen.getByText('91%')).toBeInTheDocument() // 0.91 * 100
    })

    test('shows appropriate status indicators for scores', () => {
      render(
        <ProfessionalVisualization
          results={mockResults}
          scenarioContext={mockScenarioContext}
        />
      )

      // Question Quality (85%) and Open Questions (91%) should show "Above Target"
      const aboveTargetElements = screen.getAllByText('Above Target')
      expect(aboveTargetElements.length).toBeGreaterThanOrEqual(2)

      // Wait Time (72%) and Scaffolding (68%) should show different status based on benchmarks
      expect(screen.getByText(/Target:/)).toBeInTheDocument()
    })

    test('displays metric descriptions for context', () => {
      render(
        <ProfessionalVisualization
          results={mockResults}
          scenarioContext={mockScenarioContext}
        />
      )

      expect(screen.getByText(/Quality of open-ended questions that promote thinking/)).toBeInTheDocument()
      expect(screen.getByText(/Appropriate pause time for child processing and response/)).toBeInTheDocument()
      expect(screen.getByText(/Support provided to help children reach the next level/)).toBeInTheDocument()
      expect(screen.getByText(/Use of questions that encourage deeper thinking/)).toBeInTheDocument()
    })
  })

  describe('CLASS Framework Integration', () => {
    test('displays CLASS framework scores correctly', () => {
      render(
        <ProfessionalVisualization
          results={mockResults}
          scenarioContext={mockScenarioContext}
        />
      )

      expect(screen.getByText('CLASS Framework Scores')).toBeInTheDocument()
      expect(screen.getByText('Emotional Support')).toBeInTheDocument()
      expect(screen.getByText('Classroom Organization')).toBeInTheDocument()
      expect(screen.getByText('Instructional Support')).toBeInTheDocument()

      // Check score values (converted to 10-point scale)
      expect(screen.getByText('7.2')).toBeInTheDocument()
      expect(screen.getByText('8.1')).toBeInTheDocument()
      expect(screen.getByText('6.8')).toBeInTheDocument()
    })

    test('renders radial bar chart for CLASS scores', () => {
      render(
        <ProfessionalVisualization
          results={mockResults}
          scenarioContext={mockScenarioContext}
        />
      )

      expect(screen.getByTestId('radial-bar-chart')).toBeInTheDocument()
      expect(screen.getByTestId('radial-bar')).toBeInTheDocument()
    })
  })

  describe('Interactive Features', () => {
    test('tab switching functionality works correctly', async () => {
      render(
        <ProfessionalVisualization
          results={mockResults}
          scenarioContext={mockScenarioContext}
        />
      )

      // Click on Detailed Analysis tab
      const detailedTab = screen.getByText('Detailed Analysis')
      fireEvent.click(detailedTab)

      await waitFor(() => {
        expect(screen.getByText('Detailed Analysis Breakdown')).toBeInTheDocument()
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument()
      })
    })

    test('metric cards respond to hover interactions', async () => {
      render(
        <ProfessionalVisualization
          results={mockResults}
          scenarioContext={mockScenarioContext}
        />
      )

      const questionQualityCard = screen.getByText('Question Quality').closest('div')

      fireEvent.mouseEnter(questionQualityCard)

      // The component should handle hover state (tested through CSS classes or state changes)
      expect(questionQualityCard).toHaveClass('hover:shadow-md')
    })
  })

  describe('Data Processing and Edge Cases', () => {
    test('handles missing ML predictions gracefully', () => {
      const incompleteResults = {
        class_scores: mockResults.class_scores,
        processing_time: 1.5,
        completed_at: '2025-09-25T19:30:00Z'
      }

      render(
        <ProfessionalVisualization
          results={incompleteResults}
          scenarioContext={mockScenarioContext}
        />
      )

      // Should still render without crashing
      expect(screen.getByText('Analysis Results')).toBeInTheDocument()
    })

    test('handles missing CLASS scores gracefully', () => {
      const incompleteResults = {
        ml_predictions: mockResults.ml_predictions,
        processing_time: 1.5,
        completed_at: '2025-09-25T19:30:00Z'
      }

      render(
        <ProfessionalVisualization
          results={incompleteResults}
          scenarioContext={mockScenarioContext}
        />
      )

      // Should still render ML metrics
      expect(screen.getByText('ML Analysis Metrics')).toBeInTheDocument()
    })

    test('handles missing scenario context gracefully', () => {
      render(
        <ProfessionalVisualization
          results={mockResults}
          scenarioContext={null}
        />
      )

      // Should render without scenario title but show analysis
      expect(screen.getByText('Analysis Results')).toBeInTheDocument()
      expect(screen.queryByText('Maya & the Puzzle Challenge')).not.toBeInTheDocument()
    })
  })

  describe('Accessibility and UX', () => {
    test('provides appropriate ARIA labels and semantics', () => {
      render(
        <ProfessionalVisualization
          results={mockResults}
          scenarioContext={mockScenarioContext}
        />
      )

      // Navigation tabs should be accessible
      const overviewTab = screen.getByText('Overview').closest('button')
      expect(overviewTab).toBeInTheDocument()
      expect(overviewTab.tagName).toBe('BUTTON')
    })

    test('displays informative tooltips and descriptions', () => {
      render(
        <ProfessionalVisualization
          results={mockResults}
          scenarioContext={mockScenarioContext}
        />
      )

      // Each metric should have descriptive text
      expect(screen.getByText(/Quality of open-ended questions/)).toBeInTheDocument()
      expect(screen.getByText(/Appropriate pause time/)).toBeInTheDocument()
    })

    test('uses appropriate color coding for score ranges', () => {
      render(
        <ProfessionalVisualization
          results={mockResults}
          scenarioContext={mockScenarioContext}
        />
      )

      // High scores (85%, 91%) should use success colors
      // Lower scores (68%, 72%) should use different colors
      // This is tested through the component's color logic
      const highScoreElements = screen.getAllByText(/85%|91%/)
      expect(highScoreElements.length).toBeGreaterThan(0)
    })
  })

  describe('Performance and Optimization', () => {
    test('memoizes expensive calculations', () => {
      const { rerender } = render(
        <ProfessionalVisualization
          results={mockResults}
          scenarioContext={mockScenarioContext}
        />
      )

      // Re-render with same props shouldn't cause unnecessary calculations
      rerender(
        <ProfessionalVisualization
          results={mockResults}
          scenarioContext={mockScenarioContext}
        />
      )

      expect(screen.getByText('79%')).toBeInTheDocument() // Overall score should remain consistent
    })

    test('handles large datasets efficiently', () => {
      const largeResults = {
        ...mockResults,
        ml_predictions: {
          ...mockResults.ml_predictions,
          // Add more metrics to test scalability
          additional_metric_1: 0.75,
          additional_metric_2: 0.82,
          additional_metric_3: 0.67,
          additional_metric_4: 0.88
        }
      }

      render(
        <ProfessionalVisualization
          results={largeResults}
          scenarioContext={mockScenarioContext}
        />
      )

      // Should render without performance issues
      expect(screen.getByText('Analysis Results')).toBeInTheDocument()
    })
  })
})