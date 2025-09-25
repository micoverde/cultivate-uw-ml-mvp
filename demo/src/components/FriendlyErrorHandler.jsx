import React from 'react';
import { AlertCircle, RefreshCw, ArrowRight, Lightbulb } from 'lucide-react';

/**
 * FriendlyErrorHandler Component
 * Issue #55: Navigate without technical assistance
 *
 * Transforms technical errors into helpful, encouraging guidance
 * for educational stakeholders using conversational language.
 */

const FriendlyErrorHandler = ({
  error,
  context = 'general',
  onRetry,
  onGetHelp,
  className = ""
}) => {
  // Transform technical errors into user-friendly messages
  const getFriendlyErrorMessage = (error, context) => {
    const errorString = typeof error === 'string' ? error.toLowerCase() :
                       error?.message?.toLowerCase() || 'something unexpected happened';

    // Input validation errors
    if (errorString.includes('required') || errorString.includes('empty') || errorString.includes('missing')) {
      if (context === 'scenario_input') {
        return {
          title: "Let's add your teaching scenario",
          message: "Please share a bit more about the classroom interaction you'd like us to analyze. A few sentences about what the teacher and student said will work perfectly!",
          suggestion: "Try describing what happened in the conversation - even a brief example helps our AI understand the context.",
          type: 'guidance'
        };
      } else {
        return {
          title: "We need a bit more information",
          message: "Please fill in the highlighted field so we can provide you with the best analysis possible.",
          suggestion: "Each piece of information helps us give you more personalized insights.",
          type: 'guidance'
        };
      }
    }

    // Network/connection errors
    if (errorString.includes('network') || errorString.includes('connection') || errorString.includes('timeout')) {
      return {
        title: "Taking a moment to process",
        message: "Our AI is working hard to analyze your scenario thoroughly. Sometimes this takes a little longer when we're providing detailed insights.",
        suggestion: "Please try again in a moment - the detailed analysis will be worth the wait!",
        type: 'processing'
      };
    }

    // Server errors
    if (errorString.includes('500') || errorString.includes('server') || errorString.includes('unavailable')) {
      return {
        title: "Our analysis system is catching up",
        message: "We're experiencing high demand for our teaching analysis tools right now, which shows how valuable educators find these insights!",
        suggestion: "Please try again in a few moments. We're working to get everything running smoothly.",
        type: 'temporary'
      };
    }

    // Format/validation errors
    if (errorString.includes('format') || errorString.includes('invalid') || errorString.includes('parse')) {
      if (context === 'scenario_input') {
        return {
          title: "Let's try a different format",
          message: "It looks like our system had trouble understanding the format of your scenario. Don't worry - this happens sometimes!",
          suggestion: "Try writing your scenario as a simple conversation: 'Teacher: [what they said], Student: [how they responded]'",
          type: 'guidance'
        };
      } else {
        return {
          title: "Let's adjust the format slightly",
          message: "The information looks good, but we need to adjust the format just a bit for our analysis system.",
          suggestion: "Please check that all required fields are filled out completely.",
          type: 'guidance'
        };
      }
    }

    // Generic fallback
    return {
      title: "Something unexpected happened",
      message: "We encountered an issue while processing your request, but this is easily fixable!",
      suggestion: "Please try your request again, or contact our support team if you continue to have trouble.",
      type: 'general'
    };
  };

  const errorInfo = getFriendlyErrorMessage(error, context);

  const getIconColor = () => {
    switch (errorInfo.type) {
      case 'guidance': return 'text-blue-500';
      case 'processing': return 'text-orange-500';
      case 'temporary': return 'text-yellow-500';
      default: return 'text-gray-500';
    }
  };

  const getBackgroundColor = () => {
    switch (errorInfo.type) {
      case 'guidance': return 'bg-blue-50 border-blue-200';
      case 'processing': return 'bg-orange-50 border-orange-200';
      case 'temporary': return 'bg-yellow-50 border-yellow-200';
      default: return 'bg-gray-50 border-gray-200';
    }
  };

  return (
    <div className={`rounded-lg border p-6 ${getBackgroundColor()} ${className}`}>
      <div className="flex items-start space-x-4">
        {/* Icon */}
        <div className="flex-shrink-0">
          {errorInfo.type === 'guidance' ? (
            <Lightbulb className={`w-6 h-6 ${getIconColor()}`} />
          ) : (
            <AlertCircle className={`w-6 h-6 ${getIconColor()}`} />
          )}
        </div>

        {/* Content */}
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            {errorInfo.title}
          </h3>

          <p className="text-gray-700 mb-3">
            {errorInfo.message}
          </p>

          {errorInfo.suggestion && (
            <div className="bg-white/70 rounded-md p-3 mb-4">
              <p className="text-sm text-gray-600">
                <strong>Suggestion:</strong> {errorInfo.suggestion}
              </p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-3">
            {onRetry && (
              <button
                onClick={onRetry}
                className="inline-flex items-center px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors font-medium text-sm"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Try Again
              </button>
            )}

            {onGetHelp && (
              <button
                onClick={onGetHelp}
                className="inline-flex items-center px-4 py-2 bg-white text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors font-medium text-sm"
              >
                Get Help
                <ArrowRight className="w-4 h-4 ml-2" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Quick error message component for inline use
export const InlineErrorMessage = ({ message, context = 'general', className = "" }) => {
  const errorInfo = new FriendlyErrorHandler({ error: message, context }).getFriendlyErrorMessage(message, context);

  return (
    <div className={`flex items-center space-x-2 text-sm text-red-600 mt-2 ${className}`}>
      <AlertCircle className="w-4 h-4 flex-shrink-0" />
      <span>{errorInfo.message}</span>
    </div>
  );
};

export default FriendlyErrorHandler;