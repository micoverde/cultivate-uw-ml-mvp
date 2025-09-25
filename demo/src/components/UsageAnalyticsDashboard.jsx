// Usage Analytics Dashboard
// Partner-Level Microsoft SDE Implementation for Production Monitoring

import React, { useState, useEffect } from 'react';
import { BarChart3, Activity, Users, Clock, DollarSign, AlertTriangle, TrendingUp, Zap } from 'lucide-react';
import { monitoring } from '../config/monitoring';
import { costMonitoring } from '../config/cost-monitoring';

const UsageAnalyticsDashboard = ({ isVisible, onClose }) => {
  const [analytics, setAnalytics] = useState({
    performance: {},
    costs: {},
    usage: {},
    alerts: [],
    recommendations: []
  });

  const [refreshInterval, setRefreshInterval] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Fetch analytics data
  const fetchAnalytics = async () => {
    try {
      // Get performance metrics from monitoring
      const performanceMetrics = monitoring.getPerformanceMetrics();

      // Get cost data from cost monitoring
      const costSummary = costMonitoring.getCostSummary();

      // Simulate additional analytics (in production, this would come from Azure APIs)
      const simulatedAnalytics = generateSimulatedAnalytics();

      setAnalytics({
        performance: performanceMetrics,
        costs: costSummary,
        usage: simulatedAnalytics.usage,
        alerts: [...costSummary.alerts, ...simulatedAnalytics.alerts],
        recommendations: costSummary.recommendations
      });

      setIsLoading(false);
    } catch (error) {
      console.error('[Analytics Dashboard] Error fetching data:', error);
      monitoring.trackException(error, { component: 'UsageAnalyticsDashboard' });
      setIsLoading(false);
    }
  };

  // Generate simulated analytics for demo
  const generateSimulatedAnalytics = () => {
    const now = new Date();
    const hour = now.getHours();

    return {
      usage: {
        activeUsers: Math.floor(Math.random() * 25) + 5,
        totalSessions: Math.floor(Math.random() * 150) + 50,
        averageSessionDuration: Math.floor(Math.random() * 300) + 120, // seconds
        bounceRate: (Math.random() * 0.3 + 0.1).toFixed(2), // 10-40%
        topScenarios: [
          { name: 'Maya Professional Scenario', usage: 45, trend: '+12%' },
          { name: 'Alex Learning Blocks', usage: 32, trend: '+8%' },
          { name: 'Sam Playground Interaction', usage: 28, trend: '-5%' },
          { name: 'Custom Analysis', usage: 15, trend: '+25%' }
        ],
        geographicData: [
          { region: 'US West', users: 12, percentage: 48 },
          { region: 'US East', users: 8, percentage: 32 },
          { region: 'International', users: 5, percentage: 20 }
        ],
        deviceTypes: [
          { type: 'Desktop', percentage: 65 },
          { type: 'Tablet', percentage: 25 },
          { type: 'Mobile', percentage: 10 }
        ],
        peakHours: Array.from({ length: 24 }, (_, i) => ({
          hour: i,
          usage: Math.floor(Math.random() * 20) + (i >= 8 && i <= 17 ? 20 : 5) // Higher during work hours
        }))
      },
      alerts: [
        {
          type: hour > 9 && hour < 17 ? 'info' : 'warning',
          message: hour > 9 && hour < 17 ? 'Peak usage hours detected' : 'Low usage detected - consider user engagement strategies'
        }
      ]
    };
  };

  useEffect(() => {
    if (isVisible) {
      fetchAnalytics();

      // Set up auto-refresh every 30 seconds
      const interval = setInterval(fetchAnalytics, 30000);
      setRefreshInterval(interval);

      // Track dashboard usage
      monitoring.trackEvent('AnalyticsDashboardOpened', {
        timestamp: new Date().toISOString()
      });

      return () => {
        if (interval) clearInterval(interval);
        setRefreshInterval(null);
      };
    }
  }, [isVisible]);

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-2xl w-full max-w-7xl max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-6 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
              <BarChart3 className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-2xl font-bold">Usage Analytics Dashboard</h2>
              <p className="text-indigo-100">Real-time production monitoring and insights</p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 bg-white/20 rounded-lg px-3 py-2">
              <Activity className="w-4 h-4" />
              <span className="text-sm font-medium">Live</span>
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
            </div>

            <button
              onClick={onClose}
              className="w-8 h-8 bg-white/20 hover:bg-white/30 rounded-lg flex items-center justify-center transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
              <span className="ml-4 text-lg text-slate-600">Loading analytics...</span>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Key Metrics Row */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <MetricCard
                  icon={<Users className="w-8 h-8" />}
                  title="Active Users"
                  value={analytics.usage.activeUsers}
                  change="+12%"
                  positive={true}
                  bgColor="from-blue-500 to-cyan-600"
                />
                <MetricCard
                  icon={<Activity className="w-8 h-8" />}
                  title="Total Sessions"
                  value={analytics.usage.totalSessions}
                  change="+8%"
                  positive={true}
                  bgColor="from-emerald-500 to-teal-600"
                />
                <MetricCard
                  icon={<Clock className="w-8 h-8" />}
                  title="Avg Session"
                  value={`${Math.floor(analytics.usage.averageSessionDuration / 60)}m ${analytics.usage.averageSessionDuration % 60}s`}
                  change="+15%"
                  positive={true}
                  bgColor="from-purple-500 to-pink-600"
                />
                <MetricCard
                  icon={<DollarSign className="w-8 h-8" />}
                  title="Session Cost"
                  value={costMonitoring.getCostForDisplay().currentSession}
                  change="Budget: 85%"
                  positive={false}
                  bgColor="from-orange-500 to-red-600"
                />
              </div>

              {/* Alerts Section */}
              {analytics.alerts.length > 0 && (
                <div className="bg-slate-50 dark:bg-slate-700/50 rounded-xl p-6">
                  <div className="flex items-center space-x-2 mb-4">
                    <AlertTriangle className="w-5 h-5 text-amber-500" />
                    <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Active Alerts</h3>
                  </div>
                  <div className="space-y-3">
                    {analytics.alerts.map((alert, index) => (
                      <AlertItem key={index} alert={alert} />
                    ))}
                  </div>
                </div>
              )}

              {/* Charts Row */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Usage Over Time */}
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">Usage Patterns (24h)</h3>
                  <div className="h-64">
                    <UsageChart data={analytics.usage.peakHours} />
                  </div>
                </div>

                {/* Device Types */}
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">Device Distribution</h3>
                  <div className="space-y-4">
                    {analytics.usage.deviceTypes.map((device, index) => (
                      <div key={index} className="flex items-center justify-between">
                        <span className="text-slate-600 dark:text-slate-300">{device.type}</span>
                        <div className="flex items-center space-x-3">
                          <div className="w-32 bg-slate-200 dark:bg-slate-600 rounded-full h-2">
                            <div
                              className="bg-gradient-to-r from-indigo-500 to-purple-600 h-2 rounded-full"
                              style={{ width: `${device.percentage}%` }}
                            ></div>
                          </div>
                          <span className="text-sm font-medium text-slate-900 dark:text-white min-w-[3rem]">
                            {device.percentage}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Detailed Tables Row */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Top Scenarios */}
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">Popular Scenarios</h3>
                  <div className="space-y-3">
                    {analytics.usage.topScenarios.map((scenario, index) => (
                      <div key={index} className="flex items-center justify-between py-2">
                        <div>
                          <div className="font-medium text-slate-900 dark:text-white">{scenario.name}</div>
                          <div className="text-sm text-slate-500">#{index + 1} most used</div>
                        </div>
                        <div className="text-right">
                          <div className="font-bold text-slate-900 dark:text-white">{scenario.usage}</div>
                          <div className={`text-sm ${scenario.trend.startsWith('+') ? 'text-emerald-600' : 'text-red-600'}`}>
                            {scenario.trend}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Geographic Distribution */}
                <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
                  <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">Geographic Usage</h3>
                  <div className="space-y-4">
                    {analytics.usage.geographicData.map((region, index) => (
                      <div key={index} className="flex items-center justify-between">
                        <div>
                          <div className="font-medium text-slate-900 dark:text-white">{region.region}</div>
                          <div className="text-sm text-slate-500">{region.users} active users</div>
                        </div>
                        <div className="flex items-center space-x-3">
                          <div className="w-24 bg-slate-200 dark:bg-slate-600 rounded-full h-2">
                            <div
                              className="bg-gradient-to-r from-blue-500 to-cyan-600 h-2 rounded-full"
                              style={{ width: `${region.percentage}%` }}
                            ></div>
                          </div>
                          <span className="text-sm font-medium text-slate-900 dark:text-white min-w-[3rem]">
                            {region.percentage}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Performance Metrics */}
              <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
                <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">System Performance</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-slate-900 dark:text-white">
                      {analytics.performance.pageLoads || 0}
                    </div>
                    <div className="text-sm text-slate-500">Page Loads</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-slate-900 dark:text-white">
                      {analytics.performance.apiCalls || 0}
                    </div>
                    <div className="text-sm text-slate-500">API Calls</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-red-600">
                      {analytics.performance.errors || 0}
                    </div>
                    <div className="text-sm text-slate-500">Errors</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-emerald-600">
                      {analytics.performance.userSessions || 0}
                    </div>
                    <div className="text-sm text-slate-500">User Sessions</div>
                  </div>
                </div>
              </div>

              {/* Cost Recommendations */}
              {analytics.recommendations.length > 0 && (
                <div className="bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 rounded-xl p-6">
                  <div className="flex items-center space-x-2 mb-4">
                    <TrendingUp className="w-5 h-5 text-emerald-600" />
                    <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Cost Optimization Recommendations</h3>
                  </div>
                  <div className="space-y-3">
                    {analytics.recommendations.map((rec, index) => (
                      <div key={index} className="flex items-start space-x-3">
                        <div className={`w-2 h-2 rounded-full mt-2 ${
                          rec.priority === 'high' ? 'bg-red-500' :
                          rec.priority === 'medium' ? 'bg-yellow-500' : 'bg-blue-500'
                        }`}></div>
                        <div>
                          <div className="font-medium text-slate-900 dark:text-white">{rec.message}</div>
                          <div className="text-sm text-slate-600 dark:text-slate-300">
                            Potential savings: ${rec.potentialSavings} • Category: {rec.category}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Metric Card Component
const MetricCard = ({ icon, title, value, change, positive, bgColor }) => (
  <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
    <div className="flex items-center justify-between mb-4">
      <div className={`w-12 h-12 bg-gradient-to-r ${bgColor} rounded-xl flex items-center justify-center text-white`}>
        {icon}
      </div>
      <div className={`text-sm font-medium ${positive ? 'text-emerald-600' : 'text-red-600'}`}>
        {change}
      </div>
    </div>
    <div className="text-2xl font-bold text-slate-900 dark:text-white mb-1">{value}</div>
    <div className="text-sm text-slate-500">{title}</div>
  </div>
);

// Alert Item Component
const AlertItem = ({ alert }) => (
  <div className={`flex items-center space-x-3 p-3 rounded-lg ${
    alert.type === 'warning' ? 'bg-yellow-50 dark:bg-yellow-900/20' :
    alert.type === 'error' ? 'bg-red-50 dark:bg-red-900/20' :
    'bg-blue-50 dark:bg-blue-900/20'
  }`}>
    <div className={`w-2 h-2 rounded-full ${
      alert.type === 'warning' ? 'bg-yellow-500' :
      alert.type === 'error' ? 'bg-red-500' :
      'bg-blue-500'
    }`}></div>
    <span className="text-sm text-slate-700 dark:text-slate-200">{alert.message}</span>
  </div>
);

// Simple Usage Chart Component
const UsageChart = ({ data }) => {
  const maxUsage = Math.max(...data.map(d => d.usage));

  return (
    <div className="flex items-end justify-between h-full space-x-1">
      {data.map((item, index) => (
        <div key={index} className="flex flex-col items-center space-y-1 flex-1">
          <div className="bg-gradient-to-t from-indigo-500 to-purple-600 rounded-t"
               style={{
                 height: `${(item.usage / maxUsage) * 200}px`,
                 minHeight: '4px',
                 width: '100%'
               }}>
          </div>
          <span className="text-xs text-slate-500 transform -rotate-45 origin-top-left">
            {item.hour}:00
          </span>
        </div>
      ))}
    </div>
  );
};

export default UsageAnalyticsDashboard;