/**
 * Vitest setup file for frontend testing
 * Configures global test environment
 */

import { vi } from 'vitest'

// Setup global DOM methods and objects for testing
global.performance = {
  now: vi.fn(() => Date.now())
}

// Setup console methods to be available in tests
global.console = {
  ...console,
  // Optionally mock console methods
  log: vi.fn(),
  error: vi.fn(),
  warn: vi.fn(),
  info: vi.fn()
}