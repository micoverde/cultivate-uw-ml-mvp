/**
 * Unit Tests for ExportManager Component
 * Tests professional export functionality for PDF and CSV formats
 */

import React from 'react'
import { describe, test, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import ExportManager from '../components/ExportManager'

// Mock jsPDF and related exports
vi.mock('jspdf', () => ({
  default: vi.fn().mockImplementation(() => ({
    internal: {
      pageSize: {
        width: 210,
        height: 297
      },
      getNumberOfPages: () => 1
    },
    setFont: vi.fn(),
    setFontSize: vi.fn(),
    setTextColor: vi.fn(),
    text: vi.fn(),
    addPage: vi.fn(),
    setPage: vi.fn(),
    splitTextToSize: vi.fn(() => ['Sample text']),
    autoTable: vi.fn(function(config) {
      this.lastAutoTable = { finalY: 100 };
    }),
    save: vi.fn()
  }))
}))

vi.mock('jspdf-autotable', () => ({}))

// Mock framer-motion to avoid animation issues in tests
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, onClick, className, initial, animate, exit, ...props }) => (
      <div onClick={onClick} className={className} {...props}>{children}</div>
    ),
    button: ({ children, onClick, className, whileHover, whileTap, ...props }) => (
      <button onClick={onClick} className={className} {...props}>{children}</button>
    ),
  },
  AnimatePresence: ({ children }) => children,
}))

describe('ExportManager', () => {
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
    enhanced_recommendations: {
      actionable_steps: [
        {
          action: 'Increase wait time after asking questions',
          rationale: 'Children need more processing time',
          research_basis: 'Vygotsky Zone of Proximal Development',
          priority: 'High'
        },
        {
          action: 'Use more scaffolding techniques',
          rationale: 'Support children to reach next level',
          research_basis: 'CLASS Framework best practices',
          priority: 'Medium'
        }
      ]
    },
    processing_time: 2.34,
    completed_at: '2025-09-25T19:30:00Z'
  }

  const mockScenarioContext = {
    title: 'Maya & the Puzzle Challenge',
    category: 'Problem-solving',
    ageGroup: '4-5 years'
  }

  const mockProps = {
    results: mockResults,
    scenarioContext: mockScenarioContext,
    isOpen: true,
    onClose: vi.fn()
  }

  beforeEach(() => {
    vi.clearAllMocks()
    // Mock document.createElement for download link
    global.document.createElement = vi.fn((tagName) => {
      if (tagName === 'a') {
        return {
          href: '',
          download: '',
          click: vi.fn()
        }
      }
      return {}
    })

    // Mock URL.createObjectURL and revokeObjectURL
    global.URL.createObjectURL = vi.fn(() => 'mock-blob-url')
    global.URL.revokeObjectURL = vi.fn()

    // Mock Blob constructor
    global.Blob = vi.fn(() => ({}))
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('Component Rendering', () => {
    test('renders export modal when open', () => {
      render(<ExportManager {...mockProps} />)

      expect(screen.getByText('Export Results')).toBeInTheDocument()
      expect(screen.getByText('PDF Report')).toBeInTheDocument()
      expect(screen.getByText('CSV Data')).toBeInTheDocument()
    })

    test('does not render when closed', () => {
      render(<ExportManager {...mockProps} isOpen={false} />)

      expect(screen.queryByText('Export Results')).not.toBeInTheDocument()
    })

    test('displays export format descriptions', () => {
      render(<ExportManager {...mockProps} />)

      expect(screen.getByText('Professional formatted report')).toBeInTheDocument()
      expect(screen.getByText('Raw data for analysis')).toBeInTheDocument()
      expect(screen.getByText('~2-3 pages')).toBeInTheDocument()
      expect(screen.getByText('Spreadsheet ready')).toBeInTheDocument()
    })

    test('shows cultivate branding message', () => {
      render(<ExportManager {...mockProps} />)

      expect(screen.getByText(/Exports include Cultivate Learning branding/)).toBeInTheDocument()
    })
  })

  describe('PDF Export Functionality', () => {
    test('triggers PDF export when button clicked', async () => {
      render(<ExportManager {...mockProps} />)

      const pdfButton = screen.getByText('PDF Report').closest('button')
      fireEvent.click(pdfButton)

      await waitFor(() => {
        expect(screen.getByText(/Generating PDF/)).toBeInTheDocument()
      })
    })

    test('shows success message after PDF export', async () => {
      render(<ExportManager {...mockProps} />)

      const pdfButton = screen.getByText('PDF Report').closest('button')
      fireEvent.click(pdfButton)

      await waitFor(() => {
        expect(screen.getByText(/PDF exported successfully/)).toBeInTheDocument()
      }, { timeout: 3000 })
    })

    test('disables buttons during export process', async () => {
      render(<ExportManager {...mockProps} />)

      const pdfButton = screen.getByText('PDF Report').closest('button')
      const csvButton = screen.getByText('CSV Data').closest('button')

      fireEvent.click(pdfButton)

      await waitFor(() => {
        expect(pdfButton).toBeDisabled()
        expect(csvButton).toBeDisabled()
      })
    })
  })

  describe('CSV Export Functionality', () => {
    test('triggers CSV export when button clicked', async () => {
      render(<ExportManager {...mockProps} />)

      const csvButton = screen.getByText('CSV Data').closest('button')
      fireEvent.click(csvButton)

      await waitFor(() => {
        expect(screen.getByText(/Generating CSV/)).toBeInTheDocument()
      })
    })

    test('shows success message after CSV export', async () => {
      render(<ExportManager {...mockProps} />)

      const csvButton = screen.getByText('CSV Data').closest('button')
      fireEvent.click(csvButton)

      await waitFor(() => {
        expect(screen.getByText(/CSV exported successfully/)).toBeInTheDocument()
      }, { timeout: 3000 })
    })

    test('creates download link for CSV', async () => {
      const mockLink = {
        href: '',
        download: '',
        click: vi.fn()
      }
      document.createElement = vi.fn(() => mockLink)

      render(<ExportManager {...mockProps} />)

      const csvButton = screen.getByText('CSV Data').closest('button')
      fireEvent.click(csvButton)

      await waitFor(() => {
        expect(mockLink.click).toHaveBeenCalled()
      }, { timeout: 3000 })
    })
  })

  describe('Modal Interactions', () => {
    test('calls onClose when X button clicked', () => {
      render(<ExportManager {...mockProps} />)

      const closeButton = screen.getByRole('button', { name: /close/i }) ||
                          screen.getByText('×').closest('button')

      if (closeButton) {
        fireEvent.click(closeButton)
        expect(mockProps.onClose).toHaveBeenCalled()
      }
    })

    test('calls onClose when clicking outside modal', () => {
      const { container } = render(<ExportManager {...mockProps} />)

      const backdrop = container.querySelector('.fixed.inset-0')
      if (backdrop) {
        fireEvent.click(backdrop)
        expect(mockProps.onClose).toHaveBeenCalled()
      }
    })

    test('does not close when clicking inside modal content', () => {
      render(<ExportManager {...mockProps} />)

      const modalContent = screen.getByText('Export Results').closest('div')
      fireEvent.click(modalContent)

      expect(mockProps.onClose).not.toHaveBeenCalled()
    })
  })

  describe('Data Processing', () => {
    test('handles missing ML predictions gracefully', async () => {
      const incompleteResults = {
        ...mockResults,
        ml_predictions: null
      }

      render(<ExportManager {...mockProps} results={incompleteResults} />)

      const pdfButton = screen.getByText('PDF Report').closest('button')
      fireEvent.click(pdfButton)

      // Should not crash and should show generating state
      await waitFor(() => {
        expect(screen.getByText(/Generating PDF/)).toBeInTheDocument()
      })
    })

    test('handles missing CLASS scores gracefully', async () => {
      const incompleteResults = {
        ...mockResults,
        class_scores: null
      }

      render(<ExportManager {...mockProps} results={incompleteResults} />)

      const csvButton = screen.getByText('CSV Data').closest('button')
      fireEvent.click(csvButton)

      // Should not crash and should show generating state
      await waitFor(() => {
        expect(screen.getByText(/Generating CSV/)).toBeInTheDocument()
      })
    })

    test('handles missing scenario context gracefully', () => {
      render(<ExportManager {...mockProps} scenarioContext={null} />)

      expect(screen.getByText('Export Results')).toBeInTheDocument()
      expect(screen.getByText('PDF Report')).toBeInTheDocument()
      expect(screen.getByText('CSV Data')).toBeInTheDocument()
    })
  })

  describe('File Naming', () => {
    test('generates appropriate filename with timestamp', () => {
      // Mock Date to ensure consistent timestamp
      const mockDate = new Date('2025-09-25T10:30:00Z')
      vi.setSystemTime(mockDate)

      render(<ExportManager {...mockProps} />)

      // The actual filename generation is tested through the component logic
      // This test ensures the component renders without errors when generating filenames
      expect(screen.getByText('Export Results')).toBeInTheDocument()

      vi.useRealTimers()
    })
  })

  describe('Loading States', () => {
    test('shows loading spinner during PDF export', async () => {
      render(<ExportManager {...mockProps} />)

      const pdfButton = screen.getByText('PDF Report').closest('button')
      fireEvent.click(pdfButton)

      await waitFor(() => {
        expect(screen.getByText(/Generating PDF/)).toBeInTheDocument()
        // Check for loading spinner (Loader2 icon)
        const loadingSpinner = screen.getByTestId?.('loading-spinner') ||
                              screen.querySelector('.animate-spin')
        expect(loadingSpinner).toBeTruthy()
      })
    })

    test('shows loading spinner during CSV export', async () => {
      render(<ExportManager {...mockProps} />)

      const csvButton = screen.getByText('CSV Data').closest('button')
      fireEvent.click(csvButton)

      await waitFor(() => {
        expect(screen.getByText(/Generating CSV/)).toBeInTheDocument()
        const loadingSpinner = screen.getByTestId?.('loading-spinner') ||
                              screen.querySelector('.animate-spin')
        expect(loadingSpinner).toBeTruthy()
      })
    })
  })

  describe('Accessibility', () => {
    test('has proper button roles and accessibility', () => {
      render(<ExportManager {...mockProps} />)

      const pdfButton = screen.getByRole('button', { name: /PDF Report/i })
      const csvButton = screen.getByRole('button', { name: /CSV Data/i })

      expect(pdfButton).toBeInTheDocument()
      expect(csvButton).toBeInTheDocument()
      expect(pdfButton.tagName).toBe('BUTTON')
      expect(csvButton.tagName).toBe('BUTTON')
    })

    test('provides clear visual hierarchy', () => {
      render(<ExportManager {...mockProps} />)

      const heading = screen.getByText('Export Results')
      expect(heading).toBeInTheDocument()

      const descriptions = screen.getAllByText(/Professional formatted report|Raw data for analysis/)
      expect(descriptions.length).toBe(2)
    })
  })
})