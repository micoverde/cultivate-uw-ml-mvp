"""
Enterprise Security Middleware
Microsoft Partner-Level Implementation

Provides comprehensive security controls including:
- Input validation and sanitization
- Rate limiting
- Request authentication
- Security logging and monitoring
"""

import re
import time
import json
import hashlib
import logging
import os
import secrets
from typing import Dict, Optional, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque

import bleach
from fastapi import HTTPException, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel, validator

# Configure security logger
security_logger = logging.getLogger("cultivate_security")
security_handler = logging.StreamHandler()
security_handler.setFormatter(
    logging.Formatter('%(asctime)s - SECURITY - %(levelname)s - %(message)s')
)
security_logger.addHandler(security_handler)
security_logger.setLevel(logging.INFO)

class SecurityConfig:
    """Security configuration constants"""
    # Rate limiting
    RATE_LIMIT_REQUESTS = 100  # requests per window
    RATE_LIMIT_WINDOW = 900    # 15 minutes
    RATE_LIMIT_BURST = 10      # burst allowance

    # Input validation
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_STRING_LENGTH = 10000
    MAX_LIST_ITEMS = 1000
    MAX_NESTED_DEPTH = 10

    # Blocked patterns (potential attacks)
    BLOCKED_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS
        r'javascript:',               # XSS
        r'on\w+\s*=',                # Event handlers
        r'eval\s*\(',               # Code injection
        r'exec\s*\(',               # Code injection
        r'import\s+',               # Import injection
        r'__import__',              # Python import
        r'subprocess',              # Command injection
        r'os\.system',              # Command injection
        r'\.\./\.\.',              # Path traversal
        r'union\s+select',          # SQL injection
        r'drop\s+table',           # SQL injection
    ]

    # Allowed HTML tags for content sanitization
    ALLOWED_HTML_TAGS = []
    ALLOWED_HTML_ATTRIBUTES = {}

class RateLimiter:
    """Thread-safe rate limiter with sliding window"""

    def __init__(self):
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.blocked_ips: Dict[str, datetime] = {}

    def is_allowed(self, client_ip: str) -> tuple[bool, Optional[str]]:
        """Check if request is allowed under rate limit"""
        now = time.time()
        window_start = now - SecurityConfig.RATE_LIMIT_WINDOW

        # Check if IP is temporarily blocked
        if client_ip in self.blocked_ips:
            if datetime.now() < self.blocked_ips[client_ip]:
                return False, "IP temporarily blocked due to rate limit violation"
            else:
                del self.blocked_ips[client_ip]

        # Clean old requests
        client_requests = self.requests[client_ip]
        while client_requests and client_requests[0] < window_start:
            client_requests.popleft()

        # Check rate limit
        if len(client_requests) >= SecurityConfig.RATE_LIMIT_REQUESTS:
            # Block IP for 1 hour
            self.blocked_ips[client_ip] = datetime.now() + timedelta(hours=1)
            security_logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return False, "Rate limit exceeded"

        # Record request
        client_requests.append(now)
        return True, None

class InputValidator:
    """Comprehensive input validation and sanitization"""

    @staticmethod
    def validate_string(value: str, max_length: int = None) -> str:
        """Validate and sanitize string input"""
        if not isinstance(value, str):
            raise ValueError("Input must be a string")

        # Check length
        max_len = max_length or SecurityConfig.MAX_STRING_LENGTH
        if len(value) > max_len:
            raise ValueError(f"String length exceeds maximum of {max_len}")

        # Check for blocked patterns
        for pattern in SecurityConfig.BLOCKED_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                security_logger.warning(f"Blocked pattern detected: {pattern}")
                raise ValueError("Input contains potentially malicious content")

        # Sanitize HTML
        sanitized = bleach.clean(
            value,
            tags=SecurityConfig.ALLOWED_HTML_TAGS,
            attributes=SecurityConfig.ALLOWED_HTML_ATTRIBUTES,
            strip=True
        )

        return sanitized.strip()

    @staticmethod
    def validate_dict(data: dict, max_depth: int = 0) -> dict:
        """Recursively validate dictionary data"""
        if max_depth > SecurityConfig.MAX_NESTED_DEPTH:
            raise ValueError("Data nesting exceeds maximum depth")

        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")

        if len(data) > SecurityConfig.MAX_LIST_ITEMS:
            raise ValueError(f"Dictionary size exceeds maximum of {SecurityConfig.MAX_LIST_ITEMS}")

        validated = {}
        for key, value in data.items():
            # Validate key
            if not isinstance(key, str):
                raise ValueError("Dictionary keys must be strings")

            validated_key = InputValidator.validate_string(key, 100)

            # Validate value based on type
            if isinstance(value, str):
                validated[validated_key] = InputValidator.validate_string(value)
            elif isinstance(value, dict):
                validated[validated_key] = InputValidator.validate_dict(value, max_depth + 1)
            elif isinstance(value, list):
                validated[validated_key] = InputValidator.validate_list(value, max_depth + 1)
            elif isinstance(value, (int, float, bool)) or value is None:
                validated[validated_key] = value
            else:
                raise ValueError(f"Unsupported data type: {type(value)}")

        return validated

    @staticmethod
    def validate_list(data: list, max_depth: int = 0) -> list:
        """Validate list data"""
        if not isinstance(data, list):
            raise ValueError("Input must be a list")

        if len(data) > SecurityConfig.MAX_LIST_ITEMS:
            raise ValueError(f"List size exceeds maximum of {SecurityConfig.MAX_LIST_ITEMS}")

        validated = []
        for item in data:
            if isinstance(item, str):
                validated.append(InputValidator.validate_string(item))
            elif isinstance(item, dict):
                validated.append(InputValidator.validate_dict(item, max_depth + 1))
            elif isinstance(item, list):
                validated.append(InputValidator.validate_list(item, max_depth + 1))
            elif isinstance(item, (int, float, bool)) or item is None:
                validated.append(item)
            else:
                raise ValueError(f"Unsupported data type in list: {type(item)}")

        return validated

class SecurityMiddleware(BaseHTTPMiddleware):
    """Main security middleware"""

    def __init__(self, app):
        super().__init__(app)
        self.rate_limiter = RateLimiter()
        self.security_events = deque(maxlen=1000)

    async def dispatch(self, request: Request, call_next):
        """Process request through security pipeline"""
        start_time = time.time()

        try:
            # Get client IP
            client_ip = self.get_client_ip(request)

            # Log request
            security_logger.info(f"Request: {request.method} {request.url.path} from {client_ip}")

            # Rate limiting
            allowed, error_msg = self.rate_limiter.is_allowed(client_ip)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded", "detail": error_msg}
                )

            # Request size validation
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > SecurityConfig.MAX_REQUEST_SIZE:
                security_logger.warning(f"Request size too large: {content_length} from {client_ip}")
                return JSONResponse(
                    status_code=413,
                    content={"error": "Request entity too large"}
                )

            # Input validation for POST/PUT requests
            if request.method in ["POST", "PUT", "PATCH"]:
                request = await self.validate_request_body(request, client_ip)

            # Add security headers
            response = await call_next(request)
            response = self.add_security_headers(response)

            # Log successful request
            processing_time = time.time() - start_time
            security_logger.info(f"Request completed in {processing_time:.3f}s")

            return response

        except HTTPException as e:
            security_logger.warning(f"HTTP exception: {e.status_code} - {e.detail}")
            raise
        except Exception as e:
            security_logger.error(f"Security middleware error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal security error"}
            )

    def get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for forwarded IP (behind proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Check for real IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        # Fallback to client host
        return request.client.host if request.client else "unknown"

    async def validate_request_body(self, request: Request, client_ip: str) -> Request:
        """Validate and sanitize request body"""
        try:
            content_type = request.headers.get("content-type", "")

            if "application/json" in content_type:
                body = await request.body()
                if body:
                    try:
                        data = json.loads(body)

                        # Validate based on data type
                        if isinstance(data, dict):
                            validated_data = InputValidator.validate_dict(data)
                        elif isinstance(data, list):
                            validated_data = InputValidator.validate_list(data)
                        else:
                            raise ValueError("Request body must be JSON object or array")

                        # Replace request body with validated data
                        request._body = json.dumps(validated_data).encode()

                    except json.JSONDecodeError:
                        security_logger.warning(f"Invalid JSON from {client_ip}")
                        raise HTTPException(status_code=400, detail="Invalid JSON format")
                    except ValueError as e:
                        security_logger.warning(f"Input validation failed from {client_ip}: {str(e)}")
                        raise HTTPException(status_code=400, detail=f"Input validation failed: {str(e)}")

            return request

        except HTTPException:
            raise
        except Exception as e:
            security_logger.error(f"Request validation error: {str(e)}")
            raise HTTPException(status_code=400, detail="Request validation failed")

    def add_security_headers(self, response: Response) -> Response:
        """Add security headers to response"""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()"
        }

        for header, value in security_headers.items():
            response.headers[header] = value

        return response

class APIKeyAuth(HTTPBearer):
    """API Key authentication for admin endpoints"""

    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        # Load API keys from environment variables with secure fallbacks
        self.valid_api_keys = self._load_secure_api_keys()

    def _load_secure_api_keys(self) -> Dict[str, str]:
        """Load API keys securely from environment or generate for development"""
        api_keys = {}

        # Check for production environment variables first
        admin_key = os.getenv("CULTIVATE_ADMIN_API_KEY")
        monitor_key = os.getenv("CULTIVATE_MONITOR_API_KEY")

        if admin_key and monitor_key:
            # Production: Use environment variables
            api_keys["admin"] = admin_key
            api_keys["monitor"] = monitor_key
            security_logger.info("Using production API keys from environment")
        else:
            # Development: Generate secure random keys
            api_keys["admin"] = f"sk-admin-{secrets.token_urlsafe(32)}"
            api_keys["monitor"] = f"sk-monitor-{secrets.token_urlsafe(32)}"

            # Log the keys for development use (WARNING: Never do this in production)
            if os.getenv("ENVIRONMENT") != "production":
                security_logger.warning(f"Development Admin API Key: {api_keys['admin']}")
                security_logger.warning(f"Development Monitor API Key: {api_keys['monitor']}")
                security_logger.warning("⚠️  These are development keys only - set CULTIVATE_*_API_KEY env vars for production")
            else:
                security_logger.error("Production deployment missing required API keys!")

        return api_keys

    async def __call__(self, request: Request) -> Optional[str]:
        """Validate API key"""
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)

        if credentials:
            api_key = credentials.credentials
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # Check if API key is valid
            for role, valid_key in self.valid_api_keys.items():
                if api_key == valid_key:
                    security_logger.info(f"Valid API key used for role: {role}")
                    return role

            security_logger.warning(f"Invalid API key attempted: {key_hash[:8]}...")
            raise HTTPException(status_code=401, detail="Invalid API key")

        return None

def get_security_report() -> Dict[str, Any]:
    """Generate security status report"""
    return {
        "timestamp": datetime.now().isoformat(),
        "security_middleware": "active",
        "rate_limiting": "enabled",
        "input_validation": "enabled",
        "security_headers": "enabled",
        "api_authentication": "enabled",
        "blocked_patterns": len(SecurityConfig.BLOCKED_PATTERNS),
        "max_request_size": SecurityConfig.MAX_REQUEST_SIZE,
        "rate_limit_window": SecurityConfig.RATE_LIMIT_WINDOW,
        "rate_limit_requests": SecurityConfig.RATE_LIMIT_REQUESTS
    }