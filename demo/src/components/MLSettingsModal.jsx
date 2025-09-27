import React, { useState, useEffect } from 'react';
import { Settings, X, Cpu, Zap, TrendingUp, Clock } from 'lucide-react';

const MLSettingsModal = ({ isOpen, onClose }) => {
  const [selectedModel, setSelectedModel] = useState(() => {
    return localStorage.getItem('ml_model') || 'classic';
  });
  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [performanceData, setPerformanceData] = useState({
    classic: { accuracy: 0.92, f1: 0.91, speed: 45, usage: 15000 },
    ensemble: { accuracy: 0.98, f1: 0.97, speed: 125, usage: 3000 }
  });

  useEffect(() => {
    if (isOpen) {
      loadPerformanceMetrics();
    }
  }, [isOpen]);

  const loadPerformanceMetrics = async () => {
    try {
      const response = await fetch('/api/v1/models/comparison');
      if (response.ok) {
        const data = await response.json();
        if (data.classic || data.ensemble) {
          setPerformanceData({
            classic: data.classic || performanceData.classic,
            ensemble: data.ensemble || performanceData.ensemble
          });
        }
      }
    } catch (error) {
      console.error('Error loading performance metrics:', error);
    }
  };

  const handleApply = async () => {
    if (selectedModel === localStorage.getItem('ml_model')) {
      onClose();
      return;
    }

    setIsLoading(true);
    setStatusMessage('Switching models...');

    try {
      const response = await fetch('/api/v1/models/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_type: selectedModel })
      });

      if (response.ok) {
        localStorage.setItem('ml_model', selectedModel);
        setStatusMessage(`✅ Successfully switched to ${selectedModel === 'ensemble' ? 'Ensemble' : 'Classic'} ML model`);
        setTimeout(() => {
          onClose();
          setStatusMessage('');
        }, 2000);
      } else {
        setStatusMessage('❌ Failed to switch model. Please try again.');
      }
    } catch (error) {
      console.error('Error switching model:', error);
      setStatusMessage('❌ Network error. Please check your connection.');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold text-gray-800 dark:text-white flex items-center gap-2">
            <Settings className="h-6 w-6" />
            ML Model Settings
          </h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            <X className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {/* Model Selection */}
          <div className="space-y-4 mb-6">
            <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-200">Select Model</h3>

            <label
              className={`flex items-start gap-4 p-4 rounded-lg border-2 cursor-pointer transition-all ${
                selectedModel === 'classic'
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
              }`}
            >
              <input
                type="radio"
                name="model"
                value="classic"
                checked={selectedModel === 'classic'}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="mt-1"
              />
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <Zap className="h-5 w-5 text-yellow-500" />
                  <span className="font-semibold text-gray-800 dark:text-white">Classic ML Model</span>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  Faster response time, proven in production
                </p>
              </div>
            </label>

            <label
              className={`flex items-start gap-4 p-4 rounded-lg border-2 cursor-pointer transition-all ${
                selectedModel === 'ensemble'
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
              }`}
            >
              <input
                type="radio"
                name="model"
                value="ensemble"
                checked={selectedModel === 'ensemble'}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="mt-1"
              />
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <Cpu className="h-5 w-5 text-purple-500" />
                  <span className="font-semibold text-gray-800 dark:text-white">Ensemble ML Model</span>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  Higher accuracy with 7-model voting system
                </p>
              </div>
            </label>
          </div>

          {/* Performance Comparison */}
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-200 mb-4">Performance Comparison</h3>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-sm font-semibold text-gray-700 dark:text-gray-300">
                    <th className="pb-3">Metric</th>
                    <th className="pb-3">Classic ML</th>
                    <th className="pb-3">Ensemble ML</th>
                  </tr>
                </thead>
                <tbody className="text-sm">
                  <tr className="border-t border-gray-200 dark:border-gray-700">
                    <td className="py-2 text-gray-600 dark:text-gray-400">
                      <div className="flex items-center gap-2">
                        <TrendingUp className="h-4 w-4" />
                        Accuracy
                      </div>
                    </td>
                    <td className="py-2 text-gray-800 dark:text-gray-200">
                      {Math.round(performanceData.classic.accuracy * 100)}%
                    </td>
                    <td className="py-2 text-gray-800 dark:text-gray-200">
                      {Math.round(performanceData.ensemble.accuracy * 100)}%
                    </td>
                  </tr>
                  <tr className="border-t border-gray-200 dark:border-gray-700">
                    <td className="py-2 text-gray-600 dark:text-gray-400">F1 Score</td>
                    <td className="py-2 text-gray-800 dark:text-gray-200">
                      {performanceData.classic.f1.toFixed(2)}
                    </td>
                    <td className="py-2 text-gray-800 dark:text-gray-200">
                      {performanceData.ensemble.f1.toFixed(2)}
                    </td>
                  </tr>
                  <tr className="border-t border-gray-200 dark:border-gray-700">
                    <td className="py-2 text-gray-600 dark:text-gray-400">
                      <div className="flex items-center gap-2">
                        <Clock className="h-4 w-4" />
                        Response Time
                      </div>
                    </td>
                    <td className="py-2 text-gray-800 dark:text-gray-200">
                      {performanceData.classic.speed}ms
                    </td>
                    <td className="py-2 text-gray-800 dark:text-gray-200">
                      {performanceData.ensemble.speed}ms
                    </td>
                  </tr>
                  <tr className="border-t border-gray-200 dark:border-gray-700">
                    <td className="py-2 text-gray-600 dark:text-gray-400">Total Predictions</td>
                    <td className="py-2 text-gray-800 dark:text-gray-200">
                      {Math.round(performanceData.classic.usage / 1000)}K
                    </td>
                    <td className="py-2 text-gray-800 dark:text-gray-200">
                      {Math.round(performanceData.ensemble.usage / 1000)}K
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Status Message */}
          {statusMessage && (
            <div className={`p-4 rounded-lg mb-4 ${
              statusMessage.includes('✅') ? 'bg-green-100 text-green-800' :
              statusMessage.includes('❌') ? 'bg-red-100 text-red-800' :
              'bg-blue-100 text-blue-800'
            }`}>
              {statusMessage}
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleApply}
              disabled={isLoading}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Applying...' : 'Apply Changes'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MLSettingsModal;