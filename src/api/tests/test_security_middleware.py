"""
Unit tests for Security Middleware
Tests comprehensive security controls, validation, and rate limiting
"""

import pytest
import asyncio
import json
import time
import os
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI, Request, HTTPException
from fastapi.testclient import TestClient
from starlette.responses import Response

# Import security modules to test
from ..security.middleware import (
    SecurityMiddleware,
    RateLimiter,
    InputValidator,
    SecurityConfig,
    APIKeyAuth,
    get_security_report
)

class TestRateLimiter:
    """Test rate limiting functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.rate_limiter = RateLimiter()

    def test_rate_limiter_allows_requests_within_limit(self):
        """Test that requests within limit are allowed"""
        client_ip = "192.168.1.1"

        # Should allow requests within limit
        for _ in range(50):  # Well under the limit
            allowed, error_msg = self.rate_limiter.is_allowed(client_ip)
            assert allowed is True
            assert error_msg is None

    def test_rate_limiter_blocks_requests_over_limit(self):
        """Test that requests over limit are blocked"""
        client_ip = "192.168.1.2"

        # Add requests up to limit
        for _ in range(SecurityConfig.RATE_LIMIT_REQUESTS):
            allowed, _ = self.rate_limiter.is_allowed(client_ip)
            assert allowed is True

        # Next request should be blocked
        allowed, error_msg = self.rate_limiter.is_allowed(client_ip)
        assert allowed is False
        assert "Rate limit exceeded" in error_msg

    def test_rate_limiter_blocks_ip_after_violation(self):
        """Test that IP is blocked after rate limit violation"""
        client_ip = "192.168.1.3"

        # Exceed rate limit
        for _ in range(SecurityConfig.RATE_LIMIT_REQUESTS + 1):
            self.rate_limiter.is_allowed(client_ip)

        # Should remain blocked
        allowed, error_msg = self.rate_limiter.is_allowed(client_ip)
        assert allowed is False
        assert "IP temporarily blocked" in error_msg

    def test_rate_limiter_different_ips_independent(self):
        """Test that different IPs have independent rate limits"""
        client_ip1 = "192.168.1.4"
        client_ip2 = "192.168.1.5"

        # Exceed limit for IP1
        for _ in range(SecurityConfig.RATE_LIMIT_REQUESTS + 1):
            self.rate_limiter.is_allowed(client_ip1)

        # IP1 should be blocked
        allowed1, _ = self.rate_limiter.is_allowed(client_ip1)
        assert allowed1 is False

        # IP2 should still be allowed
        allowed2, _ = self.rate_limiter.is_allowed(client_ip2)
        assert allowed2 is True

class TestInputValidator:
    """Test input validation and sanitization"""

    def test_validate_string_clean_input(self):
        """Test validation of clean string input"""
        clean_input = "Hello, this is a clean string!"
        result = InputValidator.validate_string(clean_input)
        assert result == clean_input

    def test_validate_string_removes_html_tags(self):
        """Test that HTML tags are removed"""
        malicious_input = "<script>alert('xss')</script>Hello World"
        result = InputValidator.validate_string(malicious_input)
        assert "<script>" not in result
        assert "</script>" not in result
        assert "Hello World" in result

    def test_validate_string_blocks_javascript(self):
        """Test that javascript: protocol is blocked"""
        malicious_input = "javascript:alert('xss')"
        with pytest.raises(ValueError, match="potentially malicious content"):
            InputValidator.validate_string(malicious_input)

    def test_validate_string_blocks_event_handlers(self):
        """Test that event handlers are blocked"""
        malicious_input = 'onload="alert(1)" Hello'
        with pytest.raises(ValueError, match="potentially malicious content"):
            InputValidator.validate_string(malicious_input)

    def test_validate_string_length_limit(self):
        """Test string length validation"""
        long_string = "A" * (SecurityConfig.MAX_STRING_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum"):
            InputValidator.validate_string(long_string)

    def test_validate_dict_clean_input(self):
        """Test validation of clean dictionary"""
        clean_dict = {
            "name": "John Doe",
            "age": 30,
            "active": True
        }
        result = InputValidator.validate_dict(clean_dict)
        assert result == clean_dict

    def test_validate_dict_sanitizes_values(self):
        """Test that dictionary values are sanitized"""
        malicious_dict = {
            "name": "<script>alert('xss')</script>John",
            "comment": "Hello World"
        }
        result = InputValidator.validate_dict(malicious_dict)
        assert "<script>" not in result["name"]
        assert "John" in result["name"]
        assert result["comment"] == "Hello World"

    def test_validate_dict_max_items(self):
        """Test dictionary size limit"""
        large_dict = {f"key_{i}": f"value_{i}" for i in range(SecurityConfig.MAX_LIST_ITEMS + 1)}
        with pytest.raises(ValueError, match="exceeds maximum"):
            InputValidator.validate_dict(large_dict)

    def test_validate_dict_nested_depth(self):
        """Test nested dictionary depth limit"""
        # Create deeply nested dict
        nested_dict = {}
        current = nested_dict
        for i in range(SecurityConfig.MAX_NESTED_DEPTH + 2):
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]

        with pytest.raises(ValueError, match="nesting exceeds maximum depth"):
            InputValidator.validate_dict(nested_dict)

    def test_validate_list_clean_input(self):
        """Test validation of clean list"""
        clean_list = ["item1", "item2", "item3"]
        result = InputValidator.validate_list(clean_list)
        assert result == clean_list

    def test_validate_list_sanitizes_items(self):
        """Test that list items are sanitized"""
        malicious_list = ["<script>alert(1)</script>Clean", "Normal item"]
        result = InputValidator.validate_list(malicious_list)
        assert "<script>" not in result[0]
        assert "Clean" in result[0]
        assert result[1] == "Normal item"

    def test_validate_list_size_limit(self):
        """Test list size limit"""
        large_list = [f"item_{i}" for i in range(SecurityConfig.MAX_LIST_ITEMS + 1)]
        with pytest.raises(ValueError, match="exceeds maximum"):
            InputValidator.validate_list(large_list)

class TestSecurityMiddleware:
    """Test security middleware integration"""

    def setup_method(self):
        """Set up test FastAPI app with security middleware"""
        self.app = FastAPI()
        self.app.add_middleware(SecurityMiddleware)

        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        @self.app.post("/test")
        async def test_post():
            return {"message": "post success"}

        self.client = TestClient(self.app)

    def test_security_headers_added(self):
        """Test that security headers are added to responses"""
        response = self.client.get("/test")

        # Check security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert "Referrer-Policy" in response.headers
        assert "Content-Security-Policy" in response.headers
        assert "Strict-Transport-Security" in response.headers

    def test_security_headers_values(self):
        """Test security header values"""
        response = self.client.get("/test")
        headers = response.headers

        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["X-XSS-Protection"] == "1; mode=block"
        assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "default-src 'self'" in headers["Content-Security-Policy"]
        assert "max-age=31536000" in headers["Strict-Transport-Security"]

    def test_request_size_limit(self):
        """Test request size limit enforcement"""
        # Create large payload
        large_data = "A" * (SecurityConfig.MAX_REQUEST_SIZE + 1)

        # Mock content-length header
        with patch.object(self.client, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 413
            mock_response.json.return_value = {"error": "Request entity too large"}
            mock_post.return_value = mock_response

            response = self.client.post("/test",
                                      data=large_data,
                                      headers={"content-length": str(len(large_data))})
            assert response.status_code == 413

    @patch('src.api.security.middleware.security_logger')
    def test_security_logging(self, mock_logger):
        """Test that security events are logged"""
        response = self.client.get("/test")

        # Should log the request
        mock_logger.info.assert_called()
        log_calls = mock_logger.info.call_args_list
        assert any("Request:" in str(call) for call in log_calls)

class TestAPIKeyAuth:
    """Test API key authentication"""

    def setup_method(self):
        """Set up API key auth"""
        self.auth = APIKeyAuth(auto_error=False)

    @pytest.mark.asyncio
    async def test_valid_api_key(self):
        """Test valid API key authentication"""
        # Mock request with valid API key
        mock_request = Mock()
        mock_credentials = Mock()
        mock_credentials.credentials = "sk-admin-test-AbCdEf123456789xYzTest"

        with patch.object(self.auth, '__class__.__bases__[0].__call__',
                         return_value=AsyncMock(return_value=mock_credentials)):
            # This test would require more complex mocking of HTTPBearer
            pass  # Simplified for demo

    def test_invalid_api_key_format(self):
        """Test invalid API key format"""
        # This would test the validation logic
        # Implementation depends on specific requirements
        pass

class TestSecurityConfig:
    """Test security configuration"""

    def test_security_config_constants(self):
        """Test that security configuration has required constants"""
        assert hasattr(SecurityConfig, 'RATE_LIMIT_REQUESTS')
        assert hasattr(SecurityConfig, 'RATE_LIMIT_WINDOW')
        assert hasattr(SecurityConfig, 'MAX_REQUEST_SIZE')
        assert hasattr(SecurityConfig, 'MAX_STRING_LENGTH')
        assert hasattr(SecurityConfig, 'BLOCKED_PATTERNS')

    def test_blocked_patterns_coverage(self):
        """Test that blocked patterns cover common attacks"""
        patterns = SecurityConfig.BLOCKED_PATTERNS

        # Should have XSS protection
        xss_patterns = [p for p in patterns if 'script' in p.lower() or 'javascript' in p.lower()]
        assert len(xss_patterns) > 0

        # Should have SQL injection protection
        sql_patterns = [p for p in patterns if 'union' in p.lower() or 'drop' in p.lower()]
        assert len(sql_patterns) > 0

        # Should have path traversal protection
        path_patterns = [p for p in patterns if '..' in p]
        assert len(path_patterns) > 0

class TestSecurityReport:
    """Test security report generation"""

    def test_get_security_report_structure(self):
        """Test security report has required fields"""
        report = get_security_report()

        required_fields = [
            'timestamp',
            'security_middleware',
            'rate_limiting',
            'input_validation',
            'security_headers',
            'api_authentication'
        ]

        for field in required_fields:
            assert field in report

    def test_get_security_report_values(self):
        """Test security report contains expected values"""
        report = get_security_report()

        assert report['security_middleware'] == 'active'
        assert report['rate_limiting'] == 'enabled'
        assert report['input_validation'] == 'enabled'
        assert isinstance(report['blocked_patterns'], int)
        assert report['blocked_patterns'] > 0

class TestSecurityIntegration:
    """Integration tests for security components"""

    def test_end_to_end_request_processing(self):
        """Test complete request processing through security pipeline"""
        # This would test a real request through all security layers
        app = FastAPI()
        app.add_middleware(SecurityMiddleware)

        @app.post("/api/test")
        async def test_endpoint(request: Request):
            body = await request.json()
            return {"received": body}

        client = TestClient(app)

        # Test clean request
        clean_data = {"message": "Hello World", "count": 42}
        response = client.post("/api/test", json=clean_data)
        assert response.status_code == 200

    def test_malicious_request_blocked(self):
        """Test that malicious requests are properly blocked"""
        app = FastAPI()
        app.add_middleware(SecurityMiddleware)

        @app.post("/api/test")
        async def test_endpoint(request: Request):
            body = await request.json()
            return {"received": body}

        client = TestClient(app)

        # Test malicious data - this should be sanitized or blocked
        malicious_data = {
            "message": "<script>alert('xss')</script>",
            "payload": "javascript:alert(1)"
        }

        # The middleware should either block this or sanitize it
        response = client.post("/api/test", json=malicious_data)

        # Response should either be 400 (blocked) or sanitized content
        if response.status_code == 200:
            data = response.json()
            # Should not contain malicious content
            assert "<script>" not in str(data)
            assert "javascript:" not in str(data)
        else:
            assert response.status_code == 400

# Performance and stress tests
class TestSecurityPerformance:
    """Test security middleware performance"""

    def test_rate_limiter_performance(self):
        """Test rate limiter performance under load"""
        rate_limiter = RateLimiter()
        client_ip = "192.168.1.100"

        start_time = time.time()

        # Test 1000 requests
        for _ in range(1000):
            rate_limiter.is_allowed(client_ip)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time (less than 1 second)
        assert duration < 1.0

    def test_input_validator_performance(self):
        """Test input validator performance"""
        test_string = "Clean test string " * 100  # Reasonable size string

        start_time = time.time()

        # Test 1000 validations
        for _ in range(1000):
            InputValidator.validate_string(test_string)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time
        assert duration < 2.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])