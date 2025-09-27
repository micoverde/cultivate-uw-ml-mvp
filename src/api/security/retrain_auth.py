#!/usr/bin/env python3
"""
Retraining Authentication Service
Implements password protection for model retraining functionality

Author: Claude (Partner-Level Microsoft SDE)
Issue: #184 - Secure Model Retraining
"""

import os
import bcrypt
import json
from datetime import datetime, timedelta
from typing import Optional, Dict
import logging
from dataclasses import dataclass, asdict

from fastapi import HTTPException, Header, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import redis

# Try to import Azure Blob service, handle if not yet available
try:
    from src.services.azure_blob_service import AzureBlobTrainingDataService
except ImportError:
    AzureBlobTrainingDataService = None

logger = logging.getLogger(__name__)

@dataclass
class AuthAttempt:
    """Record of authentication attempt."""
    timestamp: str
    client_ip: str
    status: str
    user_agent: Optional[str] = None
    attempts_remaining: Optional[int] = None

class RetrainAuthService:
    """
    Password-based authentication service for model retraining.

    Features:
    - Bcrypt password hashing
    - Rate limiting with Redis
    - IP-based lockout after failed attempts
    - Audit logging to Azure Blob Storage
    """

    def __init__(self):
        """Initialize authentication service."""
        # Get password from environment or use default 'lev' (hashed)
        password_to_hash = os.getenv('RETRAIN_PASSWORD', 'lev')

        # Generate bcrypt hash
        self.password_hash = bcrypt.hashpw(
            password_to_hash.encode('utf-8'),
            bcrypt.gensalt(rounds=12)
        )

        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', '6379'))
        self.redis_db = int(os.getenv('REDIS_DB', '0'))

        # Rate limiting configuration
        self.max_attempts = int(os.getenv('MAX_RETRAIN_ATTEMPTS', '3'))
        self.lockout_duration = int(os.getenv('RETRAIN_LOCKOUT_HOURS', '1')) * 3600

        # Initialize Redis client (handle connection errors gracefully)
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis connection established for rate limiting")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory rate limiting: {e}")
            self.redis_client = None
            self.redis_available = False
            # Fallback to in-memory storage
            self.memory_attempts = {}
            self.memory_lockouts = {}

        # Azure Blob service for audit logging
        if AzureBlobTrainingDataService:
            try:
                self.blob_service = AzureBlobTrainingDataService()
            except Exception as e:
                logger.warning(f"Azure Blob service not available for audit logging: {e}")
                self.blob_service = None
        else:
            self.blob_service = None

    async def validate_password(
        self,
        password: str,
        client_ip: str,
        user_agent: Optional[str] = None
    ) -> Dict:
        """
        Validate password with rate limiting.

        Args:
            password: Plain text password to validate
            client_ip: Client IP address for rate limiting
            user_agent: Optional user agent string

        Returns:
            Dict with validation result and metadata

        Raises:
            HTTPException: On authentication failure or lockout
        """
        # Check if client is locked out
        if await self._is_locked_out(client_ip):
            await self._log_auth_event(
                AuthAttempt(
                    timestamp=datetime.utcnow().isoformat(),
                    client_ip=client_ip,
                    status='locked_out',
                    user_agent=user_agent
                )
            )

            raise HTTPException(
                status_code=429,
                detail="Too many failed attempts. Account locked for 1 hour.",
                headers={"Retry-After": str(self.lockout_duration)}
            )

        # Validate password
        is_valid = bcrypt.checkpw(
            password.encode('utf-8'),
            self.password_hash
        )

        if not is_valid:
            # Track failed attempt
            attempts = await self._increment_failed_attempts(client_ip)
            remaining = max(0, self.max_attempts - attempts)

            # Lock out if max attempts reached
            if attempts >= self.max_attempts:
                await self._set_lockout(client_ip)

                await self._log_auth_event(
                    AuthAttempt(
                        timestamp=datetime.utcnow().isoformat(),
                        client_ip=client_ip,
                        status='lockout_triggered',
                        user_agent=user_agent,
                        attempts_remaining=0
                    )
                )

                raise HTTPException(
                    status_code=429,
                    detail=f"Maximum attempts exceeded. Account locked for {self.lockout_duration // 3600} hour(s).",
                    headers={"Retry-After": str(self.lockout_duration)}
                )

            # Log failed attempt
            await self._log_auth_event(
                AuthAttempt(
                    timestamp=datetime.utcnow().isoformat(),
                    client_ip=client_ip,
                    status='failed',
                    user_agent=user_agent,
                    attempts_remaining=remaining
                )
            )

            raise HTTPException(
                status_code=401,
                detail=f"Invalid password. {remaining} attempt(s) remaining.",
                headers={"X-Attempts-Remaining": str(remaining)}
            )

        # Success - clear failed attempts
        await self._clear_failed_attempts(client_ip)

        # Generate session token
        session_token = self._generate_session_token(client_ip)

        # Log successful authentication
        await self._log_auth_event(
            AuthAttempt(
                timestamp=datetime.utcnow().isoformat(),
                client_ip=client_ip,
                status='success',
                user_agent=user_agent
            )
        )

        return {
            'authenticated': True,
            'session_token': session_token,
            'expires_at': (datetime.utcnow() + timedelta(hours=1)).isoformat(),
            'client_ip': client_ip
        }

    async def validate_session_token(
        self,
        token: str,
        client_ip: str
    ) -> bool:
        """
        Validate a session token.

        Args:
            token: Session token to validate
            client_ip: Client IP to verify

        Returns:
            True if valid, False otherwise
        """
        if self.redis_available:
            stored_ip = self.redis_client.get(f"retrain_session:{token}")
            return stored_ip == client_ip
        else:
            # Fallback to memory storage
            stored_data = self.memory_attempts.get(f"session:{token}")
            if stored_data:
                return stored_data['ip'] == client_ip and \
                       datetime.fromisoformat(stored_data['expires']) > datetime.utcnow()
            return False

    async def _is_locked_out(self, client_ip: str) -> bool:
        """Check if client IP is locked out."""
        if self.redis_available:
            lockout_key = f"retrain_lockout:{client_ip}"
            return bool(self.redis_client.exists(lockout_key))
        else:
            # Fallback to memory storage
            lockout_data = self.memory_lockouts.get(client_ip)
            if lockout_data:
                if datetime.fromisoformat(lockout_data['expires']) > datetime.utcnow():
                    return True
                else:
                    # Lockout expired, remove it
                    del self.memory_lockouts[client_ip]
            return False

    async def _increment_failed_attempts(self, client_ip: str) -> int:
        """Increment and return failed attempt count."""
        if self.redis_available:
            attempt_key = f"retrain_attempts:{client_ip}"
            attempts = self.redis_client.incr(attempt_key)
            self.redis_client.expire(attempt_key, self.lockout_duration)
            return attempts
        else:
            # Fallback to memory storage
            if client_ip not in self.memory_attempts:
                self.memory_attempts[client_ip] = {
                    'count': 0,
                    'expires': (datetime.utcnow() + timedelta(seconds=self.lockout_duration)).isoformat()
                }
            self.memory_attempts[client_ip]['count'] += 1
            return self.memory_attempts[client_ip]['count']

    async def _set_lockout(self, client_ip: str):
        """Set lockout for client IP."""
        if self.redis_available:
            lockout_key = f"retrain_lockout:{client_ip}"
            self.redis_client.setex(lockout_key, self.lockout_duration, "locked")
        else:
            # Fallback to memory storage
            self.memory_lockouts[client_ip] = {
                'locked': True,
                'expires': (datetime.utcnow() + timedelta(seconds=self.lockout_duration)).isoformat()
            }

    async def _clear_failed_attempts(self, client_ip: str):
        """Clear failed attempts for client IP."""
        if self.redis_available:
            attempt_key = f"retrain_attempts:{client_ip}"
            self.redis_client.delete(attempt_key)
        else:
            # Fallback to memory storage
            if client_ip in self.memory_attempts:
                del self.memory_attempts[client_ip]

    def _generate_session_token(self, client_ip: str) -> str:
        """Generate a session token for authenticated client."""
        import secrets
        token = secrets.token_urlsafe(32)

        if self.redis_available:
            # Store in Redis with 1 hour expiry
            session_key = f"retrain_session:{token}"
            self.redis_client.setex(session_key, 3600, client_ip)
        else:
            # Fallback to memory storage
            self.memory_attempts[f"session:{token}"] = {
                'ip': client_ip,
                'expires': (datetime.utcnow() + timedelta(hours=1)).isoformat()
            }

        return token

    async def _log_auth_event(self, attempt: AuthAttempt):
        """
        Log authentication event for audit trail.

        Args:
            attempt: Authentication attempt details
        """
        try:
            # Log to console
            if attempt.status == 'success':
                logger.info(f"Successful retrain auth from {attempt.client_ip}")
            elif attempt.status == 'failed':
                logger.warning(f"Failed retrain auth from {attempt.client_ip} - {attempt.attempts_remaining} attempts remaining")
            elif attempt.status in ['locked_out', 'lockout_triggered']:
                logger.warning(f"Lockout for {attempt.client_ip}: {attempt.status}")

            # Log to Azure Blob if available
            if self.blob_service:
                try:
                    container_client = self.blob_service.blob_service.get_container_client("audit-logs")

                    timestamp = datetime.utcnow()
                    date_partition = timestamp.strftime("%Y-%m")

                    audit_event = {
                        'event_type': 'retrain_authentication',
                        'timestamp': attempt.timestamp,
                        'client_ip': attempt.client_ip,
                        'status': attempt.status,
                        'user_agent': attempt.user_agent,
                        'attempts_remaining': attempt.attempts_remaining
                    }

                    blob_name = f"retraining-events/{date_partition}/auth_{timestamp.strftime('%Y%m%d_%H%M%S')}_{attempt.client_ip.replace('.', '_')}.json"
                    blob_client = container_client.get_blob_client(blob_name)

                    blob_client.upload_blob(
                        json.dumps(audit_event, indent=2),
                        overwrite=True
                    )
                except Exception as e:
                    logger.error(f"Failed to log auth event to Azure Blob: {e}")

            # Also log to local file as backup
            log_dir = "logs/auth"
            os.makedirs(log_dir, exist_ok=True)

            log_file = os.path.join(
                log_dir,
                f"retrain_auth_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
            )

            with open(log_file, 'a') as f:
                f.write(json.dumps(asdict(attempt)) + '\n')

        except Exception as e:
            logger.error(f"Failed to log auth event: {e}")

    async def get_auth_statistics(self) -> Dict:
        """Get authentication statistics for monitoring."""
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'active_lockouts': 0,
            'recent_attempts': 0,
            'recent_successes': 0,
            'recent_failures': 0
        }

        try:
            if self.redis_available:
                # Count active lockouts
                for key in self.redis_client.scan_iter("retrain_lockout:*"):
                    stats['active_lockouts'] += 1

                # Count recent attempts (from logs)
                # This would require parsing recent log entries
            else:
                # Use memory storage
                stats['active_lockouts'] = len(self.memory_lockouts)

        except Exception as e:
            logger.error(f"Failed to get auth statistics: {e}")

        return stats

# Singleton instance
_auth_service = None

def get_retrain_auth_service() -> RetrainAuthService:
    """Get singleton instance of retrain auth service."""
    global _auth_service
    if _auth_service is None:
        _auth_service = RetrainAuthService()
    return _auth_service

# Export for easy import
__all__ = ['RetrainAuthService', 'get_retrain_auth_service', 'AuthAttempt']