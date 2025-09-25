import React, { useState } from 'react';
import { motion } from 'framer-motion';
import jsPDF from 'jspdf';
import 'jspdf-autotable';
import {
  Download,
  FileDown,
  FileSpreadsheet,
  CheckCircle,
  AlertCircle,
  Loader2,
  X
} from 'lucide-react';

/**
 * ExportManager Component
 * Professional export functionality for analysis results with Microsoft Fluent Design
 * Supports PDF reports and CSV data exports with Cultivate Learning branding
 */
const ExportManager = ({ results, scenarioContext, isOpen, onClose }) => {
  const [isExporting, setIsExporting] = useState(false);
  const [exportStatus, setExportStatus] = useState(null);
  const [exportType, setExportType] = useState(null);

  // Generate filename with timestamp
  const generateFilename = (type) => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const scenarioName = scenarioContext?.title?.replace(/[^a-zA-Z0-9]/g, '_') || 'Analysis';
    return `CultivateML_${scenarioName}_${timestamp}.${type}`;
  };

  // Format score for display
  const formatScore = (score, isPercentage = true) => {
    if (typeof score !== 'number') return 'N/A';
    return isPercentage ? `${Math.round(score * 100)}%` : score.toFixed(1);
  };

  // Export to PDF with professional formatting
  const exportToPDF = async () => {
    try {
      setIsExporting(true);
      setExportType('PDF');
      setExportStatus('generating');

      const doc = new jsPDF();
      const pageWidth = doc.internal.pageSize.width;
      const margin = 20;
      let yPosition = 20;

      // Add Cultivate Learning branding header
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(20);
      doc.setTextColor(46, 139, 87); // Cultivate Green
      doc.text('Cultivate Learning ML Analysis Report', margin, yPosition);
      yPosition += 15;

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(12);
      doc.setTextColor(100, 100, 100);
      doc.text(`Generated: ${new Date().toLocaleString()}`, margin, yPosition);
      yPosition += 10;

      if (scenarioContext) {
        doc.text(`Scenario: ${scenarioContext.title}`, margin, yPosition);
        yPosition += 6;
        doc.text(`Category: ${scenarioContext.category} | Age Group: ${scenarioContext.ageGroup}`, margin, yPosition);
        yPosition += 15;
      }

      // Executive Summary
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(16);
      doc.setTextColor(0, 0, 0);
      doc.text('Executive Summary', margin, yPosition);
      yPosition += 10;

      if (results.ml_predictions) {
        const mlScores = Object.values(results.ml_predictions);
        const avgScore = mlScores.reduce((sum, score) => sum + score, 0) / mlScores.length;

        doc.setFont('helvetica', 'normal');
        doc.setFontSize(11);
        doc.text(`Overall Quality Score: ${formatScore(avgScore)}`, margin, yPosition);
        yPosition += 6;
        doc.text(`Processing Time: ${results.processing_time?.toFixed(2)}s`, margin, yPosition);
        yPosition += 15;
      }

      // ML Analysis Metrics Table
      if (results.ml_predictions) {
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(14);
        doc.text('ML Analysis Metrics', margin, yPosition);
        yPosition += 10;

        const mlData = [
          ['Metric', 'Score', 'Assessment'],
          ['Question Quality', formatScore(results.ml_predictions.question_quality), results.ml_predictions.question_quality >= 0.7 ? 'Above Target' : 'Below Target'],
          ['Wait Time Appropriate', formatScore(results.ml_predictions.wait_time_appropriate), results.ml_predictions.wait_time_appropriate >= 0.65 ? 'Above Target' : 'Below Target'],
          ['Scaffolding Present', formatScore(results.ml_predictions.scaffolding_present), results.ml_predictions.scaffolding_present >= 0.75 ? 'Above Target' : 'Below Target'],
          ['Open-Ended Questions', formatScore(results.ml_predictions.open_ended_questions), results.ml_predictions.open_ended_questions >= 0.6 ? 'Above Target' : 'Below Target']
        ];

        doc.autoTable({
          head: [mlData[0]],
          body: mlData.slice(1),
          startY: yPosition,
          theme: 'grid',
          headStyles: { fillColor: [46, 139, 87], textColor: 255 },
          margin: { left: margin, right: margin }
        });
        yPosition = doc.lastAutoTable.finalY + 15;
      }

      // CLASS Framework Scores
      if (results.class_scores) {
        // Check if we need a new page
        if (yPosition > 200) {
          doc.addPage();
          yPosition = 20;
        }

        doc.setFont('helvetica', 'bold');
        doc.setFontSize(14);
        doc.text('CLASS Framework Assessment', margin, yPosition);
        yPosition += 10;

        const classData = [
          ['Domain', 'Score (0-10)', 'Rating'],
          ['Emotional Support', results.class_scores.emotional_support.toFixed(1), results.class_scores.emotional_support >= 7 ? 'Strong' : results.class_scores.emotional_support >= 5 ? 'Adequate' : 'Needs Improvement'],
          ['Classroom Organization', results.class_scores.classroom_organization.toFixed(1), results.class_scores.classroom_organization >= 7 ? 'Strong' : results.class_scores.classroom_organization >= 5 ? 'Adequate' : 'Needs Improvement'],
          ['Instructional Support', results.class_scores.instructional_support.toFixed(1), results.class_scores.instructional_support >= 7 ? 'Strong' : results.class_scores.instructional_support >= 5 ? 'Adequate' : 'Needs Improvement']
        ];

        doc.autoTable({
          head: [classData[0]],
          body: classData.slice(1),
          startY: yPosition,
          theme: 'grid',
          headStyles: { fillColor: [65, 105, 225], textColor: 255 },
          margin: { left: margin, right: margin }
        });
        yPosition = doc.lastAutoTable.finalY + 15;
      }

      // Enhanced Recommendations
      if (results.enhanced_recommendations && results.enhanced_recommendations.actionable_steps) {
        // Check if we need a new page
        if (yPosition > 220) {
          doc.addPage();
          yPosition = 20;
        }

        doc.setFont('helvetica', 'bold');
        doc.setFontSize(14);
        doc.text('Key Recommendations', margin, yPosition);
        yPosition += 10;

        doc.setFont('helvetica', 'normal');
        doc.setFontSize(10);

        results.enhanced_recommendations.actionable_steps.slice(0, 5).forEach((step, index) => {
          if (yPosition > 250) {
            doc.addPage();
            yPosition = 20;
          }

          doc.text(`${index + 1}. ${step.action}`, margin, yPosition);
          yPosition += 6;

          if (step.rationale) {
            const rationale = doc.splitTextToSize(`   Rationale: ${step.rationale}`, pageWidth - 2 * margin);
            doc.text(rationale, margin, yPosition);
            yPosition += rationale.length * 4 + 3;
          }
        });
      }

      // Footer with branding
      const pageCount = doc.internal.getNumberOfPages();
      for (let i = 1; i <= pageCount; i++) {
        doc.setPage(i);
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(8);
        doc.setTextColor(150, 150, 150);
        doc.text('Generated with Cultivate Learning ML Platform', margin, doc.internal.pageSize.height - 10);
        doc.text(`Page ${i} of ${pageCount}`, pageWidth - margin - 20, doc.internal.pageSize.height - 10);
      }

      // Save the PDF
      doc.save(generateFilename('pdf'));

      setExportStatus('success');
      setTimeout(() => {
        setExportStatus(null);
        setIsExporting(false);
      }, 2000);

    } catch (error) {
      console.error('PDF export error:', error);
      setExportStatus('error');
      setTimeout(() => {
        setExportStatus(null);
        setIsExporting(false);
      }, 3000);
    }
  };

  // Export to CSV with structured data
  const exportToCSV = async () => {
    try {
      setIsExporting(true);
      setExportType('CSV');
      setExportStatus('generating');

      const csvData = [];

      // Header information
      csvData.push(['Cultivate Learning ML Analysis Export']);
      csvData.push(['Generated', new Date().toISOString()]);
      if (scenarioContext) {
        csvData.push(['Scenario', scenarioContext.title]);
        csvData.push(['Category', scenarioContext.category]);
        csvData.push(['Age Group', scenarioContext.ageGroup]);
      }
      csvData.push(['Processing Time (seconds)', results.processing_time]);
      csvData.push([]); // Empty row

      // ML Predictions
      if (results.ml_predictions) {
        csvData.push(['ML ANALYSIS METRICS']);
        csvData.push(['Metric', 'Score', 'Percentage', 'Status']);

        const metrics = [
          ['Question Quality', results.ml_predictions.question_quality, formatScore(results.ml_predictions.question_quality), results.ml_predictions.question_quality >= 0.7 ? 'Above Target' : 'Below Target'],
          ['Wait Time Appropriate', results.ml_predictions.wait_time_appropriate, formatScore(results.ml_predictions.wait_time_appropriate), results.ml_predictions.wait_time_appropriate >= 0.65 ? 'Above Target' : 'Below Target'],
          ['Scaffolding Present', results.ml_predictions.scaffolding_present, formatScore(results.ml_predictions.scaffolding_present), results.ml_predictions.scaffolding_present >= 0.75 ? 'Above Target' : 'Below Target'],
          ['Open-Ended Questions', results.ml_predictions.open_ended_questions, formatScore(results.ml_predictions.open_ended_questions), results.ml_predictions.open_ended_questions >= 0.6 ? 'Above Target' : 'Below Target']
        ];

        metrics.forEach(metric => csvData.push(metric));
        csvData.push([]); // Empty row
      }

      // CLASS Scores
      if (results.class_scores) {
        csvData.push(['CLASS FRAMEWORK SCORES']);
        csvData.push(['Domain', 'Score (0-10)', 'Rating']);

        const classMetrics = [
          ['Emotional Support', results.class_scores.emotional_support.toFixed(2), results.class_scores.emotional_support >= 7 ? 'Strong' : results.class_scores.emotional_support >= 5 ? 'Adequate' : 'Needs Improvement'],
          ['Classroom Organization', results.class_scores.classroom_organization.toFixed(2), results.class_scores.classroom_organization >= 7 ? 'Strong' : results.class_scores.classroom_organization >= 5 ? 'Adequate' : 'Needs Improvement'],
          ['Instructional Support', results.class_scores.instructional_support.toFixed(2), results.class_scores.instructional_support >= 7 ? 'Strong' : results.class_scores.instructional_support >= 5 ? 'Adequate' : 'Needs Improvement']
        ];

        classMetrics.forEach(metric => csvData.push(metric));
        csvData.push([]); // Empty row
      }

      // Recommendations
      if (results.enhanced_recommendations?.actionable_steps) {
        csvData.push(['RECOMMENDATIONS']);
        csvData.push(['Priority', 'Action', 'Rationale', 'Research Basis']);

        results.enhanced_recommendations.actionable_steps.forEach((step, index) => {
          csvData.push([
            step.priority || (index + 1),
            step.action || '',
            step.rationale || '',
            step.research_basis || ''
          ]);
        });
      }

      // Convert to CSV format
      const csvContent = csvData.map(row =>
        row.map(field =>
          typeof field === 'string' && field.includes(',')
            ? `"${field.replace(/"/g, '""')}"`
            : field
        ).join(',')
      ).join('\n');

      // Create and download CSV file
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = generateFilename('csv');
      link.click();
      URL.revokeObjectURL(link.href);

      setExportStatus('success');
      setTimeout(() => {
        setExportStatus(null);
        setIsExporting(false);
      }, 2000);

    } catch (error) {
      console.error('CSV export error:', error);
      setExportStatus('error');
      setTimeout(() => {
        setExportStatus(null);
        setIsExporting(false);
      }, 3000);
    }
  };

  if (!isOpen) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 md:flex md:items-center md:justify-center md:bg-black md:bg-opacity-50"
      onClick={onClose}
    >
      <motion.div
        initial={{ y: '100%', opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        exit={{ y: '100%', opacity: 0 }}
        transition={{ type: 'spring', damping: 25, stiffness: 500 }}
        className="bg-white w-full h-full md:max-w-md md:w-full md:h-auto md:rounded-xl md:shadow-2xl md:mx-4 overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Mobile-Optimized Header */}
        <div className="sticky top-0 bg-white border-b border-gray-200 md:border-0 mobile-padding md:p-8 md:pb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Download className="w-6 h-6 text-indigo-600 mr-3" />
              <h2 className="mobile-text-xl md:text-2xl font-bold text-gray-900">Export Results</h2>
            </div>
            <button
              onClick={onClose}
              className="touch-feedback mobile-nav-button md:text-gray-400 md:hover:text-gray-600 md:transition-colors"
              aria-label="Close export dialog"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Main Content Container */}
        <div className="mobile-padding md:px-8 md:pb-8">
          {/* Status Messages */}
          {exportStatus && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className={`mb-4 md:mb-6 mobile-padding md:p-4 rounded-lg flex items-center ${
                exportStatus === 'success'
                  ? 'bg-green-100 text-green-800'
                  : exportStatus === 'error'
                  ? 'bg-red-100 text-red-800'
                  : 'bg-blue-100 text-blue-800'
              }`}
            >
              {exportStatus === 'generating' && <Loader2 className="w-5 h-5 mr-2 animate-spin" />}
              {exportStatus === 'success' && <CheckCircle className="w-5 h-5 mr-2" />}
              {exportStatus === 'error' && <AlertCircle className="w-5 h-5 mr-2" />}

              <span className="mobile-text-sm md:text-base">
                {exportStatus === 'generating' && `Generating ${exportType} report...`}
                {exportStatus === 'success' && `${exportType} exported successfully!`}
                {exportStatus === 'error' && `Failed to export ${exportType}. Please try again.`}
              </span>
            </motion.div>
          )}

          {/* Export Options */}
          <div className="space-y-4">
            <p className="text-gray-600 mb-4 md:mb-6 mobile-text-sm md:text-base">
              Choose your preferred export format for the analysis results.
            </p>

            {/* PDF Export Button - Mobile Optimized */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={exportToPDF}
              disabled={isExporting}
              className="w-full touch-target-lg flex items-center justify-between mobile-padding md:p-4 bg-gradient-to-r from-red-500 to-red-600 text-white rounded-lg hover:from-red-600 hover:to-red-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="flex items-center min-w-0 flex-1">
                <FileDown className="w-5 h-5 mr-3 flex-shrink-0" />
                <div className="text-left min-w-0">
                  <div className="font-semibold mobile-text-base md:text-base">PDF Report</div>
                  <div className="mobile-text-sm md:text-sm opacity-90 truncate">Professional formatted report</div>
                </div>
              </div>
              <div className="flex-shrink-0 ml-3">
                {isExporting && exportType === 'PDF' ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <div className="mobile-text-xs md:text-sm opacity-75">~2-3 pages</div>
                )}
              </div>
            </motion.button>

            {/* CSV Export Button - Mobile Optimized */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={exportToCSV}
              disabled={isExporting}
              className="w-full touch-target-lg flex items-center justify-between mobile-padding md:p-4 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg hover:from-green-600 hover:to-green-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="flex items-center min-w-0 flex-1">
                <FileSpreadsheet className="w-5 h-5 mr-3 flex-shrink-0" />
                <div className="text-left min-w-0">
                  <div className="font-semibold mobile-text-base md:text-base">CSV Data</div>
                  <div className="mobile-text-sm md:text-sm opacity-90 truncate">Raw data for analysis</div>
                </div>
              </div>
              <div className="flex-shrink-0 ml-3">
                {isExporting && exportType === 'CSV' ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <div className="mobile-text-xs md:text-sm opacity-75">Spreadsheet ready</div>
                )}
              </div>
            </motion.button>
          </div>

          {/* Footer Info - Mobile Optimized */}
          <div className="mt-6 pt-4 border-t border-gray-200">
            <p className="mobile-text-xs md:text-sm text-gray-500 text-center">
              Exports include Cultivate Learning branding and are suitable for stakeholder sharing
            </p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default ExportManager;