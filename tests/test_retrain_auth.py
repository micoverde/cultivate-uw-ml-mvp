#!/usr/bin/env python3
"""
Unit tests for Retrain Authentication Service
Tests password protection and rate limiting functionality

Author: Claude (Partner-Level Microsoft SDE)
Issue: #184 - Secure Model Retraining
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import asyncio
import bcrypt
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.security.retrain_auth import (
    RetrainAuthService,
    AuthAttempt,
    get_retrain_auth_service
)


class TestRetrainAuthService(unittest.TestCase):
    """Test suite for Retrain Authentication Service."""

    def setUp(self):
        """Set up test fixtures."""
        self.auth_service = None
        self.test_password = "lev"
        self.test_client_ip = "192.168.1.1"
        self.test_user_agent = "TestAgent/1.0"

    @patch('src.api.security.retrain_auth.redis.Redis')
    def test_service_initialization_with_redis(self, mock_redis):
        """Test service initialization with Redis available."""
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        service = RetrainAuthService()

        self.assertTrue(service.redis_available)
        self.assertIsNotNone(service.redis_client)
        mock_redis_instance.ping.assert_called_once()

    @patch('src.api.security.retrain_auth.redis.Redis')
    def test_service_initialization_without_redis(self, mock_redis):
        """Test service initialization when Redis is not available."""
        mock_redis.side_effect = Exception("Redis connection failed")

        service = RetrainAuthService()

        self.assertFalse(service.redis_available)
        self.assertIsNone(service.redis_client)
        self.assertEqual(service.memory_attempts, {})
        self.assertEqual(service.memory_lockouts, {})

    @patch('src.api.security.retrain_auth.redis.Redis')
    async def test_validate_password_success(self, mock_redis):
        """Test successful password validation."""
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.exists.return_value = False
        mock_redis_instance.delete.return_value = True
        mock_redis.return_value = mock_redis_instance

        service = RetrainAuthService()

        # Mock the blob service to avoid actual Azure calls
        service.blob_service = None

        result = await service.validate_password(
            password=self.test_password,
            client_ip=self.test_client_ip,
            user_agent=self.test_user_agent
        )

        self.assertTrue(result['authenticated'])
        self.assertIn('session_token', result)
        self.assertIn('expires_at', result)
        self.assertEqual(result['client_ip'], self.test_client_ip)

    @patch('src.api.security.retrain_auth.redis.Redis')
    async def test_validate_password_failure(self, mock_redis):
        """Test failed password validation."""
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.exists.return_value = False
        mock_redis_instance.incr.return_value = 1
        mock_redis_instance.expire.return_value = True
        mock_redis.return_value = mock_redis_instance

        service = RetrainAuthService()
        service.blob_service = None

        with self.assertRaises(Exception) as context:
            await service.validate_password(
                password="wrong_password",
                client_ip=self.test_client_ip,
                user_agent=self.test_user_agent
            )

        self.assertIn("Invalid password", str(context.exception))

    @patch('src.api.security.retrain_auth.redis.Redis')
    async def test_rate_limiting_lockout(self, mock_redis):
        """Test rate limiting and lockout after max attempts."""
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.exists.return_value = False
        mock_redis_instance.expire.return_value = True
        mock_redis.return_value = mock_redis_instance

        service = RetrainAuthService()
        service.blob_service = None
        service.max_attempts = 3

        # Simulate 3 failed attempts
        mock_redis_instance.incr.side_effect = [1, 2, 3]

        for i in range(3):
            try:
                await service.validate_password(
                    password="wrong_password",
                    client_ip=self.test_client_ip,
                    user_agent=self.test_user_agent
                )
            except Exception as e:
                if i < 2:
                    self.assertIn(f"{2-i} attempt(s) remaining", str(e))
                else:
                    self.assertIn("Maximum attempts exceeded", str(e))

        # Verify lockout was set
        mock_redis_instance.setex.assert_called_once()

    @patch('src.api.security.retrain_auth.redis.Redis')
    async def test_is_locked_out_with_redis(self, mock_redis):
        """Test lockout check with Redis."""
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.exists.return_value = True
        mock_redis.return_value = mock_redis_instance

        service = RetrainAuthService()

        is_locked = await service._is_locked_out(self.test_client_ip)

        self.assertTrue(is_locked)
        mock_redis_instance.exists.assert_called_with(f"retrain_lockout:{self.test_client_ip}")

    async def test_is_locked_out_memory_fallback(self):
        """Test lockout check with memory fallback."""
        service = RetrainAuthService()
        service.redis_available = False
        service.memory_lockouts = {
            self.test_client_ip: {
                'locked': True,
                'expires': (datetime.utcnow() + timedelta(hours=1)).isoformat()
            }
        }

        is_locked = await service._is_locked_out(self.test_client_ip)

        self.assertTrue(is_locked)

    async def test_is_locked_out_expired_memory(self):
        """Test expired lockout removal from memory."""
        service = RetrainAuthService()
        service.redis_available = False
        service.memory_lockouts = {
            self.test_client_ip: {
                'locked': True,
                'expires': (datetime.utcnow() - timedelta(hours=1)).isoformat()
            }
        }

        is_locked = await service._is_locked_out(self.test_client_ip)

        self.assertFalse(is_locked)
        self.assertNotIn(self.test_client_ip, service.memory_lockouts)

    @patch('src.api.security.retrain_auth.redis.Redis')
    async def test_increment_failed_attempts_redis(self, mock_redis):
        """Test incrementing failed attempts with Redis."""
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.incr.return_value = 2
        mock_redis_instance.expire.return_value = True
        mock_redis.return_value = mock_redis_instance

        service = RetrainAuthService()

        attempts = await service._increment_failed_attempts(self.test_client_ip)

        self.assertEqual(attempts, 2)
        mock_redis_instance.incr.assert_called_with(f"retrain_attempts:{self.test_client_ip}")

    async def test_increment_failed_attempts_memory(self):
        """Test incrementing failed attempts with memory fallback."""
        service = RetrainAuthService()
        service.redis_available = False

        attempts1 = await service._increment_failed_attempts(self.test_client_ip)
        attempts2 = await service._increment_failed_attempts(self.test_client_ip)

        self.assertEqual(attempts1, 1)
        self.assertEqual(attempts2, 2)
        self.assertEqual(service.memory_attempts[self.test_client_ip]['count'], 2)

    @patch('src.api.security.retrain_auth.secrets')
    @patch('src.api.security.retrain_auth.redis.Redis')
    def test_generate_session_token_redis(self, mock_redis, mock_secrets):
        """Test session token generation with Redis."""
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.setex.return_value = True
        mock_redis.return_value = mock_redis_instance

        mock_secrets.token_urlsafe.return_value = "test_token_12345"

        service = RetrainAuthService()

        token = service._generate_session_token(self.test_client_ip)

        self.assertEqual(token, "test_token_12345")
        mock_redis_instance.setex.assert_called_with(
            "retrain_session:test_token_12345",
            3600,
            self.test_client_ip
        )

    @patch('src.api.security.retrain_auth.secrets')
    def test_generate_session_token_memory(self, mock_secrets):
        """Test session token generation with memory fallback."""
        mock_secrets.token_urlsafe.return_value = "test_token_67890"

        service = RetrainAuthService()
        service.redis_available = False

        token = service._generate_session_token(self.test_client_ip)

        self.assertEqual(token, "test_token_67890")
        self.assertIn(f"session:test_token_67890", service.memory_attempts)

    @patch('src.api.security.retrain_auth.redis.Redis')
    async def test_validate_session_token_redis(self, mock_redis):
        """Test session token validation with Redis."""
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.get.return_value = self.test_client_ip
        mock_redis.return_value = mock_redis_instance

        service = RetrainAuthService()

        is_valid = await service.validate_session_token("valid_token", self.test_client_ip)

        self.assertTrue(is_valid)
        mock_redis_instance.get.assert_called_with("retrain_session:valid_token")

    async def test_validate_session_token_memory(self):
        """Test session token validation with memory fallback."""
        service = RetrainAuthService()
        service.redis_available = False
        service.memory_attempts["session:test_token"] = {
            'ip': self.test_client_ip,
            'expires': (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }

        is_valid = await service.validate_session_token("test_token", self.test_client_ip)

        self.assertTrue(is_valid)

    @patch('builtins.open', create=True)
    @patch('os.makedirs')
    async def test_log_auth_event_success(self, mock_makedirs, mock_open):
        """Test logging authentication events."""
        service = RetrainAuthService()
        service.blob_service = None

        attempt = AuthAttempt(
            timestamp=datetime.utcnow().isoformat(),
            client_ip=self.test_client_ip,
            status='success',
            user_agent=self.test_user_agent
        )

        await service._log_auth_event(attempt)

        mock_makedirs.assert_called_with("logs/auth", exist_ok=True)
        mock_open.assert_called_once()

    async def test_get_auth_statistics_redis(self):
        """Test getting authentication statistics with Redis."""
        service = RetrainAuthService()
        service.redis_available = True

        # Mock Redis scan
        service.redis_client = MagicMock()
        service.redis_client.scan_iter.return_value = [
            "retrain_lockout:192.168.1.1",
            "retrain_lockout:192.168.1.2"
        ]

        stats = await service.get_auth_statistics()

        self.assertEqual(stats['active_lockouts'], 2)
        self.assertIn('timestamp', stats)

    async def test_get_auth_statistics_memory(self):
        """Test getting authentication statistics with memory fallback."""
        service = RetrainAuthService()
        service.redis_available = False
        service.memory_lockouts = {
            '192.168.1.1': {'locked': True},
            '192.168.1.2': {'locked': True}
        }

        stats = await service.get_auth_statistics()

        self.assertEqual(stats['active_lockouts'], 2)


class TestAuthAttempt(unittest.TestCase):
    """Test suite for AuthAttempt dataclass."""

    def test_auth_attempt_creation(self):
        """Test AuthAttempt creation with all fields."""
        attempt = AuthAttempt(
            timestamp="2025-01-27T10:30:00",
            client_ip="192.168.1.1",
            status="success",
            user_agent="Mozilla/5.0",
            attempts_remaining=2
        )

        self.assertEqual(attempt.timestamp, "2025-01-27T10:30:00")
        self.assertEqual(attempt.client_ip, "192.168.1.1")
        self.assertEqual(attempt.status, "success")
        self.assertEqual(attempt.user_agent, "Mozilla/5.0")
        self.assertEqual(attempt.attempts_remaining, 2)

    def test_auth_attempt_optional_fields(self):
        """Test AuthAttempt with optional fields."""
        attempt = AuthAttempt(
            timestamp="2025-01-27T10:30:00",
            client_ip="192.168.1.1",
            status="failed"
        )

        self.assertIsNone(attempt.user_agent)
        self.assertIsNone(attempt.attempts_remaining)


class TestPasswordHashing(unittest.TestCase):
    """Test suite for password hashing functionality."""

    def test_password_hash_verification(self):
        """Test bcrypt password hashing and verification."""
        password = "lev"
        wrong_password = "wrong"

        # Generate hash
        hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12))

        # Verify correct password
        self.assertTrue(bcrypt.checkpw(password.encode('utf-8'), hash))

        # Verify wrong password fails
        self.assertFalse(bcrypt.checkpw(wrong_password.encode('utf-8'), hash))

    def test_password_hash_uniqueness(self):
        """Test that same password generates different hashes."""
        password = "lev"

        hash1 = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        hash2 = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Hashes should be different due to different salts
        self.assertNotEqual(hash1, hash2)

        # But both should verify the same password
        self.assertTrue(bcrypt.checkpw(password.encode('utf-8'), hash1))
        self.assertTrue(bcrypt.checkpw(password.encode('utf-8'), hash2))


class TestSingletonPattern(unittest.TestCase):
    """Test singleton pattern for auth service."""

    @patch('src.api.security.retrain_auth.redis.Redis')
    def test_singleton_instance(self, mock_redis):
        """Test that get_retrain_auth_service returns singleton."""
        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        service1 = get_retrain_auth_service()
        service2 = get_retrain_auth_service()

        self.assertIs(service1, service2)


def run_async_test(coro):
    """Helper to run async tests."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == '__main__':
    unittest.main()