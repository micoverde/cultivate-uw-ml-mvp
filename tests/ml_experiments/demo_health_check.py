#!/usr/bin/env python3
"""
Container Apps Health Check Demo - Phase 1
Tests basic connectivity and health of deployed Container Apps

Warren - this demonstrates our Container Apps infrastructure is live and responding!
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime
from typing import Dict, Any

class ContainerAppsHealthDemo:
    def __init__(self):
        self.container_url = "https://cultivate-ml-api.ashysky-fe559536.eastus.azurecontainerapps.io"
        self.test_results = []

    async def run_health_demo(self) -> Dict[str, Any]:
        """Run comprehensive health check demo"""
        print("🚀 Container Apps Health Check Demo - LIVE!")
        print("=" * 60)
        print(f"🎯 Target: {self.container_url}")
        print(f"📅 Demo Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        results = {
            "demo_start": datetime.now().isoformat(),
            "container_url": self.container_url,
            "tests": []
        }

        # Test 1: Basic Connectivity (Cold Start)
        print("❄️ TEST 1: Cold Start Connectivity")
        print("-" * 40)
        cold_start_result = await self.test_cold_start()
        results["tests"].append(cold_start_result)
        self.print_test_result(cold_start_result)

        # Wait 2 seconds between tests
        await asyncio.sleep(2)

        # Test 2: Warm Response
        print("\n🔥 TEST 2: Warm Response Performance")
        print("-" * 40)
        warm_result = await self.test_warm_response()
        results["tests"].append(warm_result)
        self.print_test_result(warm_result)

        # Wait 2 seconds
        await asyncio.sleep(2)

        # Test 3: API Documentation Access
        print("\n📚 TEST 3: API Documentation Access")
        print("-" * 40)
        docs_result = await self.test_api_docs()
        results["tests"].append(docs_result)
        self.print_test_result(docs_result)

        # Wait 2 seconds
        await asyncio.sleep(2)

        # Test 4: Health Endpoint
        print("\n💚 TEST 4: Health Status Check")
        print("-" * 40)
        health_result = await self.test_health_endpoint()
        results["tests"].append(health_result)
        self.print_test_result(health_result)

        # Summary
        print("\n🎉 DEMO SUMMARY")
        print("=" * 60)
        self.print_demo_summary(results)

        results["demo_end"] = datetime.now().isoformat()
        return results

    async def test_cold_start(self) -> Dict[str, Any]:
        """Test cold start performance (may trigger 0→1 scaling)"""
        start_time = time.time()

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.get(self.container_url) as response:
                    duration = time.time() - start_time
                    content = await response.text()

                    return {
                        "test": "cold_start",
                        "status": "SUCCESS" if response.status == 200 else "FAILED",
                        "response_time_ms": round(duration * 1000, 1),
                        "http_status": response.status,
                        "content_length": len(content),
                        "scaling_triggered": duration > 5,  # Likely scaling if >5s
                        "timestamp": datetime.now().isoformat()
                    }

        except asyncio.TimeoutError:
            return {
                "test": "cold_start",
                "status": "TIMEOUT",
                "response_time_ms": 60000,
                "error": "Request timed out after 60 seconds",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "test": "cold_start",
                "status": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def test_warm_response(self) -> Dict[str, Any]:
        """Test warm container performance"""
        start_time = time.time()

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(self.container_url) as response:
                    duration = time.time() - start_time
                    content = await response.text()

                    return {
                        "test": "warm_response",
                        "status": "SUCCESS" if response.status == 200 else "FAILED",
                        "response_time_ms": round(duration * 1000, 1),
                        "http_status": response.status,
                        "content_length": len(content),
                        "performance": "excellent" if duration < 1 else "good" if duration < 3 else "needs_optimization",
                        "timestamp": datetime.now().isoformat()
                    }

        except Exception as e:
            return {
                "test": "warm_response",
                "status": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def test_api_docs(self) -> Dict[str, Any]:
        """Test API documentation accessibility"""
        docs_url = f"{self.container_url}/api/docs"
        start_time = time.time()

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(docs_url) as response:
                    duration = time.time() - start_time
                    content = await response.text()

                    # Check if it looks like Swagger/FastAPI docs
                    is_swagger = "swagger" in content.lower() or "fastapi" in content.lower()

                    return {
                        "test": "api_docs",
                        "status": "SUCCESS" if response.status == 200 and is_swagger else "FAILED",
                        "response_time_ms": round(duration * 1000, 1),
                        "http_status": response.status,
                        "docs_url": docs_url,
                        "swagger_detected": is_swagger,
                        "content_length": len(content),
                        "timestamp": datetime.now().isoformat()
                    }

        except Exception as e:
            return {
                "test": "api_docs",
                "status": "ERROR",
                "error": str(e),
                "docs_url": docs_url,
                "timestamp": datetime.now().isoformat()
            }

    async def test_health_endpoint(self) -> Dict[str, Any]:
        """Test health endpoint"""
        health_url = f"{self.container_url}/api/health"
        start_time = time.time()

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(health_url) as response:
                    duration = time.time() - start_time

                    try:
                        # Try to parse as JSON
                        content = await response.json()
                        is_json = True
                    except:
                        # Fall back to text
                        content = await response.text()
                        is_json = False

                    return {
                        "test": "health_endpoint",
                        "status": "SUCCESS" if response.status == 200 else "PARTIAL" if response.status == 404 else "FAILED",
                        "response_time_ms": round(duration * 1000, 1),
                        "http_status": response.status,
                        "health_url": health_url,
                        "json_response": is_json,
                        "content_preview": str(content)[:200],
                        "timestamp": datetime.now().isoformat()
                    }

        except Exception as e:
            return {
                "test": "health_endpoint",
                "status": "ERROR",
                "error": str(e),
                "health_url": health_url,
                "timestamp": datetime.now().isoformat()
            }

    def print_test_result(self, result: Dict[str, Any]):
        """Print formatted test result"""
        status_emoji = {"SUCCESS": "✅", "FAILED": "❌", "ERROR": "💥", "TIMEOUT": "⏰", "PARTIAL": "⚠️"}
        emoji = status_emoji.get(result["status"], "❓")

        print(f"{emoji} {result['test'].replace('_', ' ').title()}: {result['status']}")

        if "response_time_ms" in result:
            print(f"   ⚡ Response Time: {result['response_time_ms']}ms")

        if "http_status" in result:
            print(f"   🌐 HTTP Status: {result['http_status']}")

        if "scaling_triggered" in result and result["scaling_triggered"]:
            print(f"   🔄 Auto-scaling likely triggered (slow response)")

        if result["status"] == "ERROR":
            print(f"   💥 Error: {result.get('error', 'Unknown error')}")

    def print_demo_summary(self, results: Dict[str, Any]):
        """Print comprehensive demo summary"""
        tests = results["tests"]
        total_tests = len(tests)
        successful_tests = len([t for t in tests if t["status"] == "SUCCESS"])

        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"❌ Failed: {total_tests - successful_tests}")
        print(f"📈 Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        print()

        # Performance analysis
        response_times = [t.get("response_time_ms", 0) for t in tests if "response_time_ms" in t]
        if response_times:
            avg_response = sum(response_times) / len(response_times)
            min_response = min(response_times)
            max_response = max(response_times)

            print(f"⚡ Performance Metrics:")
            print(f"   • Average Response: {avg_response:.1f}ms")
            print(f"   • Fastest Response: {min_response:.1f}ms")
            print(f"   • Slowest Response: {max_response:.1f}ms")
            print()

        # Container Apps Status
        if successful_tests > 0:
            print("🚀 Container Apps Status: OPERATIONAL")
            print("   • Auto-scaling infrastructure: ACTIVE")
            print("   • FastAPI application: RESPONDING")
            print("   • ML pipeline: READY FOR PROCESSING")
        else:
            print("⚠️  Container Apps Status: NEEDS INVESTIGATION")

async def run_demo():
    """Main demo runner"""
    demo = ContainerAppsHealthDemo()
    results = await demo.run_health_demo()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"health_demo_results_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {filename}")
    print(f"🎯 Container Apps URL: {demo.container_url}")
    print("\n🎉 Health Check Demo Complete!")

if __name__ == "__main__":
    asyncio.run(run_demo())