import React from 'react';
import { ChevronRight, ArrowLeft } from 'lucide-react';

/**
 * NavigationContext Component
 * Issue #55: Navigate without technical assistance
 *
 * Provides clear navigation context, progress indicators, and next action guidance
 * for non-technical educational stakeholders.
 */

const NavigationContext = ({
  currentStep,
  totalSteps,
  stepLabels = [],
  nextActionText,
  onNext,
  onBack,
  canProceed = true,
  showProgress = true,
  className = ""
}) => {
  const getStepStatus = (stepIndex) => {
    if (stepIndex < currentStep) return 'completed';
    if (stepIndex === currentStep) return 'current';
    return 'upcoming';
  };

  const getStepIcon = (stepIndex, status) => {
    if (status === 'completed') {
      return (
        <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
          <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
          </svg>
        </div>
      );
    } else if (status === 'current') {
      return (
        <div className="w-8 h-8 bg-indigo-600 rounded-full flex items-center justify-center animate-pulse">
          <span className="text-white font-semibold text-sm">{stepIndex + 1}</span>
        </div>
      );
    } else {
      return (
        <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">
          <span className="text-gray-500 font-medium text-sm">{stepIndex + 1}</span>
        </div>
      );
    }
  };

  return (
    <div className={`bg-white border-b border-gray-200 ${className}`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        {/* Progress Indicators */}
        {showProgress && (
          <div className="flex items-center justify-center mb-4 space-x-2 sm:space-x-4">
            {Array.from({ length: totalSteps }, (_, index) => {
              const status = getStepStatus(index);
              return (
                <React.Fragment key={index}>
                  <div className="flex flex-col items-center">
                    {getStepIcon(index, status)}
                    {stepLabels[index] && (
                      <span className={`mt-2 text-xs font-medium ${
                        status === 'current' ? 'text-indigo-600' :
                        status === 'completed' ? 'text-green-600' : 'text-gray-500'
                      }`}>
                        {stepLabels[index]}
                      </span>
                    )}
                  </div>
                  {index < totalSteps - 1 && (
                    <ChevronRight className={`w-4 h-4 mx-2 ${
                      index < currentStep ? 'text-green-500' : 'text-gray-300'
                    }`} />
                  )}
                </React.Fragment>
              );
            })}
          </div>
        )}

        {/* Navigation Actions */}
        <div className="flex justify-between items-center">
          {/* Back Button */}
          <button
            onClick={onBack}
            className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-600 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 hover:text-gray-700 transition-colors touch-target"
            disabled={currentStep === 0}
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            <span className="hidden sm:inline">Back</span>
            <span className="sm:hidden">←</span>
          </button>

          {/* Current Step Context */}
          <div className="text-center flex-1 mx-4">
            <p className="text-sm text-gray-600">
              Step {currentStep + 1} of {totalSteps}
              {stepLabels[currentStep] && (
                <span className="block sm:inline sm:ml-2 font-medium text-gray-900">
                  {stepLabels[currentStep]}
                </span>
              )}
            </p>
          </div>

          {/* Next Action Button */}
          {nextActionText && (
            <button
              onClick={onNext}
              disabled={!canProceed}
              className={`inline-flex items-center px-6 py-2 text-sm font-semibold rounded-lg transition-all touch-target ${
                canProceed
                  ? 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-sm hover:shadow-md'
                  : 'bg-gray-200 text-gray-400 cursor-not-allowed'
              }`}
            >
              {nextActionText}
              <ChevronRight className="w-4 h-4 ml-2" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default NavigationContext;