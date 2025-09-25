// API Configuration
const API_CONFIG = {
  // Use environment variable if available, fallback to relative paths for proxy
  BASE_URL: import.meta.env.VITE_API_BASE_URL || '',
  // Production detection
  // Security: Use endsWith for safer hostname validation to prevent subdomain attacks
  IS_PRODUCTION: import.meta.env.VITE_ENVIRONMENT === 'production' ||
                 window.location.hostname.endsWith('.azurestaticapps.net') ||
                 window.location.hostname === 'azurestaticapps.net',
  ENDPOINTS: {
    ANALYZE_TRANSCRIPT: '/api/v1/analyze/transcript',
    ANALYZE_STATUS: '/api/v1/analyze/status',
    ANALYZE_RESULTS: '/api/v1/analyze/results',
    HEALTH: '/api/health'
  }
};

// Helper function to build full API URL
export const buildApiUrl = (endpoint) => {
  // If we have a base URL from env, use it directly
  if (API_CONFIG.BASE_URL) {
    return `${API_CONFIG.BASE_URL}${endpoint}`;
  }
  // Otherwise use relative path (for dev proxy)
  return endpoint;
};

// Export individual endpoints for convenience
export const API_ENDPOINTS = {
  ANALYZE_TRANSCRIPT: buildApiUrl(API_CONFIG.ENDPOINTS.ANALYZE_TRANSCRIPT),
  ANALYZE_STATUS: (id) => buildApiUrl(`${API_CONFIG.ENDPOINTS.ANALYZE_STATUS}/${id}`),
  ANALYZE_RESULTS: (id) => buildApiUrl(`${API_CONFIG.ENDPOINTS.ANALYZE_RESULTS}/${id}`),
  HEALTH: buildApiUrl(API_CONFIG.ENDPOINTS.HEALTH)
};

export default API_CONFIG;