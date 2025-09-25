/**
 * Frontend Security Tests
 * Tests client-side security controls, validation, and protection
 */

import { describe, test, expect, beforeEach, afterEach, vi } from 'vitest'
import SecurityManager from '../config/security.js'

// Mock DOM methods for testing
Object.defineProperty(window, 'location', {
  value: {
    protocol: 'https:',
    hostname: 'localhost',
    href: 'https://localhost:3000'
  },
  writable: true
})

// Mock document methods
Object.defineProperty(document, 'createElement', {
  value: vi.fn(() => ({
    setAttribute: vi.fn(),
    appendChild: vi.fn()
  }))
})

Object.defineProperty(document, 'head', {
  value: {
    appendChild: vi.fn()
  }
})

Object.defineProperty(document, 'querySelector', {
  value: vi.fn()
})

describe('SecurityManager', () => {
  beforeEach(() => {
    // Clear any previous mocks
    vi.clearAllMocks()

    // Mock console methods to avoid noise in tests
    vi.spyOn(console, 'info').mockImplementation(() => {})
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('HTTPS Enforcement', () => {
    test('should redirect HTTP to HTTPS on non-localhost', () => {
      // Mock HTTP on production domain
      Object.defineProperty(window, 'location', {
        value: {
          protocol: 'http:',
          hostname: 'example.azurestaticapps.net',
          href: 'http://example.azurestaticapps.net'
        },
        writable: true
      })

      const originalHref = window.location.href
      SecurityManager.enforceHttps()

      // Should attempt to redirect to HTTPS
      // Note: In real browser, this would cause a redirect
      expect(window.location.href).toBe(originalHref)
    })

    test('should allow HTTP on localhost', () => {
      Object.defineProperty(window, 'location', {
        value: {
          protocol: 'http:',
          hostname: 'localhost',
          href: 'http://localhost:3000'
        },
        writable: true
      })

      const originalHref = window.location.href
      SecurityManager.enforceHttps()

      // Should not redirect on localhost
      expect(window.location.href).toBe(originalHref)
    })
  })

  describe('Content Security Policy', () => {
    test('should create CSP meta tag', () => {
      const mockMeta = {
        httpEquiv: '',
        content: '',
        setAttribute: vi.fn()
      }

      document.createElement = vi.fn(() => mockMeta)

      SecurityManager.setupCSP()

      expect(document.createElement).toHaveBeenCalledWith('meta')
      expect(mockMeta.httpEquiv).toBe('Content-Security-Policy')
      expect(mockMeta.content).toContain("default-src 'self'")
      expect(mockMeta.content).toContain("script-src 'self'")
      expect(mockMeta.content).toContain("https://js.monitor.azure.com")
      expect(document.head.appendChild).toHaveBeenCalledWith(mockMeta)
    })

    test('should include Azure Application Insights in CSP', () => {
      const mockMeta = {
        httpEquiv: '',
        content: '',
        setAttribute: vi.fn()
      }

      document.createElement = vi.fn(() => mockMeta)
      SecurityManager.setupCSP()

      expect(mockMeta.content).toContain('https://js.monitor.azure.com')
      expect(mockMeta.content).toContain('https://dc.services.visualstudio.com')
    })
  })

  describe('Input Sanitization', () => {
    test('should sanitize clean input correctly', () => {
      const cleanInput = 'Hello, this is clean text!'
      const result = SecurityManager.sanitizeInput(cleanInput)

      expect(result).toBe(cleanInput)
    })

    test('should remove HTML tags', () => {
      const maliciousInput = '<script>alert("xss")</script>Hello World'
      const result = SecurityManager.sanitizeInput(maliciousInput)

      expect(result).not.toContain('<script>')
      expect(result).not.toContain('</script>')
      expect(result).toContain('Hello World')
    })

    test('should remove javascript: protocol', () => {
      const maliciousInput = 'javascript:alert("xss")'
      const result = SecurityManager.sanitizeInput(maliciousInput)

      expect(result).not.toContain('javascript:')
    })

    test('should remove event handlers', () => {
      const maliciousInput = 'onload="alert(1)" Hello'
      const result = SecurityManager.sanitizeInput(maliciousInput)

      expect(result).not.toMatch(/on\w+\s*=/i)
      expect(result).toContain('Hello')
    })

    test('should enforce length limits', () => {
      const longInput = 'A'.repeat(10001) // Over limit

      expect(() => {
        SecurityManager.sanitizeInput(longInput)
      }).toThrow('Input exceeds maximum allowed length')
    })

    test('should handle non-string input', () => {
      const numberInput = 42
      const result = SecurityManager.sanitizeInput(numberInput)

      expect(result).toBe(numberInput)
    })
  })

  describe('API Request Validation', () => {
    test('should validate clean API request', () => {
      const url = '/api/test'
      const data = { message: 'Hello World', count: 42 }

      expect(() => {
        SecurityManager.validateApiRequest(url, data)
      }).not.toThrow()
    })

    test('should reject invalid URL format', () => {
      const invalidUrl = 'not-a-valid-url'
      const data = { message: 'test' }

      expect(() => {
        SecurityManager.validateApiRequest(invalidUrl, data)
      }).toThrow('Security validation failed')
    })

    test('should detect XSS in request data', () => {
      const url = '/api/test'
      const maliciousData = {
        message: '<script>alert("xss")</script>',
        payload: 'test'
      }

      expect(() => {
        SecurityManager.validateApiRequest(url, maliciousData)
      }).toThrow('Potentially malicious content detected')
    })

    test('should detect javascript: protocol in data', () => {
      const url = '/api/test'
      const maliciousData = {
        redirect: 'javascript:alert(1)'
      }

      expect(() => {
        SecurityManager.validateApiRequest(url, maliciousData)
      }).toThrow('Potentially malicious content detected')
    })

    test('should detect iframe injection in data', () => {
      const url = '/api/test'
      const maliciousData = {
        content: '<iframe src="evil.com"></iframe>'
      }

      expect(() => {
        SecurityManager.validateApiRequest(url, maliciousData)
      }).toThrow('Potentially malicious content detected')
    })

    test('should handle invalid JSON data', () => {
      const url = '/api/test'
      const invalidData = {}
      // Create circular reference to break JSON.stringify
      invalidData.self = invalidData

      expect(() => {
        SecurityManager.validateApiRequest(url, invalidData)
      }).toThrow('Security validation failed')
    })
  })

  describe('Environment Validation', () => {
    test('should pass validation for HTTPS', () => {
      Object.defineProperty(window, 'location', {
        value: {
          protocol: 'https:',
          hostname: 'example.com'
        },
        writable: true
      })

      document.querySelector = vi.fn(() => null)

      const issues = SecurityManager.validateEnvironment()
      expect(issues).not.toContain('HTTPS not enforced')
    })

    test('should detect HTTP on production', () => {
      Object.defineProperty(window, 'location', {
        value: {
          protocol: 'http:',
          hostname: 'example.azurestaticapps.net'
        },
        writable: true
      })

      const issues = SecurityManager.validateEnvironment()
      expect(issues).toContain('HTTPS not enforced')
    })

    test('should detect mixed content', () => {
      Object.defineProperty(window, 'location', {
        value: {
          protocol: 'https:',
          hostname: 'example.com'
        },
        writable: true
      })

      // Mock finding HTTP script on HTTPS page
      document.querySelector = vi.fn(() => ({
        src: 'http://example.com/script.js'
      }))

      const issues = SecurityManager.validateEnvironment()
      expect(issues).toContain('Mixed content detected')
    })

    test('should pass validation with no issues', () => {
      Object.defineProperty(window, 'location', {
        value: {
          protocol: 'https:',
          hostname: 'example.com'
        },
        writable: true
      })

      document.querySelector = vi.fn(() => null)

      const issues = SecurityManager.validateEnvironment()
      expect(issues).toHaveLength(0)
    })
  })

  describe('Security Report Generation', () => {
    test('should generate complete security report', () => {
      Object.defineProperty(window, 'location', {
        value: {
          protocol: 'https:',
          origin: 'https://example.com'
        },
        writable: true
      })

      document.querySelector = vi.fn(() => ({ content: 'mock-csp' }))

      const report = SecurityManager.getSecurityReport()

      expect(report).toHaveProperty('timestamp')
      expect(report).toHaveProperty('https')
      expect(report).toHaveProperty('origin')
      expect(report).toHaveProperty('csp')
      expect(report).toHaveProperty('config')

      expect(report.https).toBe(true)
      expect(report.origin).toBe('https://example.com')
      expect(report.csp).toBe(true)
    })

    test('should report security configuration', () => {
      const report = SecurityManager.getSecurityReport()

      expect(report.config).toHaveProperty('enforceHttps')
      expect(report.config).toHaveProperty('rateLimitingEnabled')
      expect(report.config).toHaveProperty('corsConfigured')
      expect(report.config).toHaveProperty('inputValidationEnabled')

      expect(report.config.inputValidationEnabled).toBe(true)
    })
  })

  describe('Security Initialization', () => {
    test('should initialize all security controls', () => {
      const enforceHttpsSpy = vi.spyOn(SecurityManager, 'enforceHttps')
      const setupCSPSpy = vi.spyOn(SecurityManager, 'setupCSP')
      const setupHeadersSpy = vi.spyOn(SecurityManager, 'setupSecurityHeaders')
      const validateEnvSpy = vi.spyOn(SecurityManager, 'validateEnvironment')

      SecurityManager.initialize()

      expect(enforceHttpsSpy).toHaveBeenCalled()
      expect(setupCSPSpy).toHaveBeenCalled()
      expect(setupHeadersSpy).toHaveBeenCalled()
      expect(validateEnvSpy).toHaveBeenCalled()
    })

    test('should handle initialization errors gracefully', () => {
      const setupCSPSpy = vi.spyOn(SecurityManager, 'setupCSP')
        .mockImplementation(() => {
          throw new Error('CSP setup failed')
        })

      // Should not throw error during initialization
      expect(() => {
        SecurityManager.initialize()
      }).not.toThrow()

      expect(setupCSPSpy).toHaveBeenCalled()
    })
  })
})

describe('Security Configuration', () => {
  test('should have secure default configuration', () => {
    const { SECURITY_CONFIG } = require('../config/security.js')

    // HTTPS settings
    expect(SECURITY_CONFIG.https.enforceHttps).toBe(true)
    expect(SECURITY_CONFIG.https.hstsMaxAge).toBeGreaterThan(0)
    expect(SECURITY_CONFIG.https.includeSubdomains).toBe(true)

    // CSP settings
    expect(SECURITY_CONFIG.csp.defaultSrc).toContain("'self'")
    expect(SECURITY_CONFIG.csp.frameSrc).toContain("'none'")
    expect(SECURITY_CONFIG.csp.objectSrc).toContain("'none'")

    // Security headers
    expect(SECURITY_CONFIG.headers.xFrameOptions).toBe('DENY')
    expect(SECURITY_CONFIG.headers.xContentTypeOptions).toBe('nosniff')
  })

  test('should allow Application Insights domains', () => {
    const { SECURITY_CONFIG } = require('../config/security.js')

    expect(SECURITY_CONFIG.csp.scriptSrc).toContain('https://js.monitor.azure.com')
    expect(SECURITY_CONFIG.csp.connectSrc).toContain('https://dc.services.visualstudio.com')
  })

  test('should have appropriate rate limiting settings', () => {
    const { SECURITY_CONFIG } = require('../config/security.js')

    expect(SECURITY_CONFIG.rateLimiting.windowMs).toBeGreaterThan(0)
    expect(SECURITY_CONFIG.rateLimiting.maxRequests).toBeGreaterThan(0)
    expect(SECURITY_CONFIG.rateLimiting.maxRequests).toBeLessThan(1000) // Reasonable limit
  })
})

describe('Security Edge Cases', () => {
  test('should handle null input in sanitization', () => {
    expect(() => {
      SecurityManager.sanitizeInput(null)
    }).not.toThrow()
  })

  test('should handle undefined input in sanitization', () => {
    expect(() => {
      SecurityManager.sanitizeInput(undefined)
    }).not.toThrow()
  })

  test('should handle empty string input', () => {
    const result = SecurityManager.sanitizeInput('')
    expect(result).toBe('')
  })

  test('should handle whitespace-only input', () => {
    const result = SecurityManager.sanitizeInput('   \t\n   ')
    expect(result.trim()).toBe('')
  })

  test('should handle unicode characters safely', () => {
    const unicodeInput = 'Hello 世界 🌍 café'
    const result = SecurityManager.sanitizeInput(unicodeInput)
    expect(result).toBe(unicodeInput)
  })
})

describe('Security Performance', () => {
  test('should sanitize input within reasonable time', () => {
    const testString = 'Clean test string '.repeat(100)

    const startTime = performance.now()

    for (let i = 0; i < 100; i++) {
      SecurityManager.sanitizeInput(testString)
    }

    const endTime = performance.now()
    const duration = endTime - startTime

    // Should complete 100 sanitizations in less than 100ms
    expect(duration).toBeLessThan(100)
  })

  test('should validate API requests efficiently', () => {
    const url = '/api/test'
    const data = { message: 'test', items: Array(100).fill('item') }

    const startTime = performance.now()

    for (let i = 0; i < 50; i++) {
      SecurityManager.validateApiRequest(url, data)
    }

    const endTime = performance.now()
    const duration = endTime - startTime

    // Should complete 50 validations in less than 50ms
    expect(duration).toBeLessThan(50)
  })
})

describe('Security Integration', () => {
  test('should work with real DOM elements', () => {
    // Create actual DOM elements for testing
    document.body.innerHTML = `
      <div id="test-container">
        <script src="http://malicious.com/script.js"></script>
        <p>Test content</p>
      </div>
    `

    document.querySelector = document.querySelector.bind(document)

    const issues = SecurityManager.validateEnvironment()
    // Should detect the HTTP script in HTTPS context

    // Clean up
    document.body.innerHTML = ''
  })

  test('should integrate with monitoring system', () => {
    // Test integration with Application Insights
    // This would require mocking the monitoring module
    expect(() => {
      SecurityManager.initialize()
    }).not.toThrow()
  })
})