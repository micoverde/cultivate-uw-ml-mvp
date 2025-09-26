#!/usr/bin/env python3
"""
Container Apps Auto-Scaling Demo - Phase 2
Demonstrates 0→3 replica auto-scaling behavior with real-time monitoring

Warren - this shows our Container Apps automatically scaling based on demand!
"""

import asyncio
import aiohttp
import subprocess
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
import concurrent.futures

class ContainerAppsAutoScalingDemo:
    def __init__(self):
        self.container_url = "https://cultivate-ml-api.ashysky-fe559536.eastus.azurecontainerapps.io"
        self.resource_group = "cultivate-ml-rg"
        self.app_name = "cultivate-ml-api"
        self.scaling_events = []

    async def run_scaling_demo(self) -> Dict[str, Any]:
        """Run comprehensive auto-scaling demonstration"""
        print("🔄 Container Apps Auto-Scaling Demo - LIVE!")
        print("=" * 60)
        print(f"🎯 Target: {self.container_url}")
        print(f"📅 Demo Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        results = {
            "demo_start": datetime.now().isoformat(),
            "container_url": self.container_url,
            "phases": []
        }

        # Phase 1: Check Initial State (Scale-to-Zero)
        print("🟦 PHASE 1: Initial State Check")
        print("-" * 40)
        initial_state = await self.check_initial_state()
        results["phases"].append(initial_state)
        self.print_phase_result(initial_state)

        await asyncio.sleep(3)

        # Phase 2: Single Request (Cold Start - 0→1)
        print("\n❄️ PHASE 2: Cold Start Single Request (0→1)")
        print("-" * 40)
        cold_start = await self.test_cold_start_scaling()
        results["phases"].append(cold_start)
        self.print_phase_result(cold_start)

        await asyncio.sleep(5)

        # Phase 3: Concurrent Load (Scale 1→2 or 1→3)
        print("\n🚀 PHASE 3: Concurrent Load Test (1→2→3)")
        print("-" * 40)
        load_test = await self.test_concurrent_scaling()
        results["phases"].append(load_test)
        self.print_phase_result(load_test)

        await asyncio.sleep(5)

        # Phase 4: Monitor Scale-Down
        print("\n📉 PHASE 4: Scale-Down Monitoring")
        print("-" * 40)
        scale_down = await self.monitor_scale_down()
        results["phases"].append(scale_down)
        self.print_phase_result(scale_down)

        # Summary
        print("\n🎉 AUTO-SCALING DEMO SUMMARY")
        print("=" * 60)
        self.print_scaling_summary(results)

        results["demo_end"] = datetime.now().isoformat()
        results["scaling_events"] = self.scaling_events
        return results

    async def get_replica_count(self) -> int:
        """Get current replica count from Azure CLI"""
        try:
            cmd = [
                "az", "containerapp", "replica", "list",
                "--name", self.app_name,
                "--resource-group", self.resource_group,
                "--query", "length(@)",
                "-o", "tsv"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                count = int(result.stdout.strip() or "0")
                return count
            else:
                return -1  # Error indicator

        except Exception:
            return -1  # Error indicator

    async def check_initial_state(self) -> Dict[str, Any]:
        """Check if Container Apps is in scale-to-zero state"""
        print("🔍 Checking initial replica count...")

        initial_replicas = await self.get_replica_count()

        # Wait a moment and check again for consistency
        await asyncio.sleep(2)
        confirmed_replicas = await self.get_replica_count()

        return {
            "phase": "initial_state",
            "status": "SUCCESS" if initial_replicas >= 0 else "ERROR",
            "initial_replicas": initial_replicas,
            "confirmed_replicas": confirmed_replicas,
            "scale_to_zero_active": initial_replicas == 0,
            "timestamp": datetime.now().isoformat()
        }

    async def test_cold_start_scaling(self) -> Dict[str, Any]:
        """Test cold start scaling from 0→1 replicas"""
        print("🥶 Triggering cold start request...")

        # Get initial replica count
        pre_request_replicas = await self.get_replica_count()

        # Make request that should trigger scaling
        start_time = time.time()

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                async with session.get(self.container_url) as response:
                    request_duration = time.time() - start_time

                    # Check replica count immediately after request
                    post_request_replicas = await self.get_replica_count()

                    # Wait 10 seconds and check again to see scaling
                    await asyncio.sleep(10)
                    stabilized_replicas = await self.get_replica_count()

                    scaling_occurred = stabilized_replicas > pre_request_replicas

                    if scaling_occurred:
                        self.scaling_events.append(f"{datetime.now().strftime('%H:%M:%S')}: Scaled {pre_request_replicas}→{stabilized_replicas}")

                    return {
                        "phase": "cold_start",
                        "status": "SUCCESS" if response.status == 200 else "FAILED",
                        "request_duration_ms": round(request_duration * 1000, 1),
                        "pre_request_replicas": pre_request_replicas,
                        "post_request_replicas": post_request_replicas,
                        "stabilized_replicas": stabilized_replicas,
                        "scaling_occurred": scaling_occurred,
                        "scaling_direction": "UP" if stabilized_replicas > pre_request_replicas else "NONE",
                        "http_status": response.status,
                        "timestamp": datetime.now().isoformat()
                    }

        except Exception as e:
            return {
                "phase": "cold_start",
                "status": "ERROR",
                "error": str(e),
                "pre_request_replicas": pre_request_replicas,
                "timestamp": datetime.now().isoformat()
            }

    async def test_concurrent_scaling(self) -> Dict[str, Any]:
        """Test concurrent load to trigger 1→2→3 scaling"""
        print("⚡ Generating concurrent load (5 simultaneous requests)...")

        pre_load_replicas = await self.get_replica_count()

        # Create 5 concurrent requests
        start_time = time.time()

        async def make_request(session, request_id):
            try:
                async with session.get(self.container_url) as response:
                    return {
                        "request_id": request_id,
                        "status": response.status,
                        "success": response.status == 200
                    }
            except Exception as e:
                return {
                    "request_id": request_id,
                    "status": 0,
                    "success": False,
                    "error": str(e)
                }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                # Fire all requests simultaneously
                tasks = [make_request(session, i) for i in range(5)]
                request_results = await asyncio.gather(*tasks)

                load_duration = time.time() - start_time

                # Check replica count immediately after load
                during_load_replicas = await self.get_replica_count()

                # Wait 15 seconds for scaling to stabilize
                print("   ⏳ Waiting for scaling to stabilize...")
                await asyncio.sleep(15)
                post_load_replicas = await self.get_replica_count()

                successful_requests = len([r for r in request_results if r["success"]])
                scaling_occurred = post_load_replicas > pre_load_replicas

                if scaling_occurred:
                    self.scaling_events.append(f"{datetime.now().strftime('%H:%M:%S')}: Load test scaled {pre_load_replicas}→{post_load_replicas}")

                return {
                    "phase": "concurrent_load",
                    "status": "SUCCESS" if successful_requests > 0 else "FAILED",
                    "load_duration_ms": round(load_duration * 1000, 1),
                    "total_requests": len(request_results),
                    "successful_requests": successful_requests,
                    "pre_load_replicas": pre_load_replicas,
                    "during_load_replicas": during_load_replicas,
                    "post_load_replicas": post_load_replicas,
                    "scaling_occurred": scaling_occurred,
                    "max_replicas_reached": post_load_replicas >= 3,
                    "request_results": request_results,
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            return {
                "phase": "concurrent_load",
                "status": "ERROR",
                "error": str(e),
                "pre_load_replicas": pre_load_replicas,
                "timestamp": datetime.now().isoformat()
            }

    async def monitor_scale_down(self) -> Dict[str, Any]:
        """Monitor scale-down behavior over time"""
        print("📊 Monitoring scale-down behavior (60 seconds)...")

        initial_replicas = await self.get_replica_count()
        replica_history = [(0, initial_replicas)]

        # Monitor for 60 seconds with 10-second intervals
        for i in range(6):
            await asyncio.sleep(10)
            current_replicas = await self.get_replica_count()
            replica_history.append(((i + 1) * 10, current_replicas))

            # Check for scale-down events
            if len(replica_history) > 1:
                prev_replicas = replica_history[-2][1]
                if current_replicas < prev_replicas:
                    self.scaling_events.append(f"{datetime.now().strftime('%H:%M:%S')}: Scaled down {prev_replicas}→{current_replicas}")

            print(f"   📈 {(i + 1) * 10}s: {current_replicas} replicas")

        final_replicas = replica_history[-1][1]
        scale_down_occurred = final_replicas < initial_replicas

        return {
            "phase": "scale_down_monitoring",
            "status": "SUCCESS",
            "monitoring_duration_seconds": 60,
            "initial_replicas": initial_replicas,
            "final_replicas": final_replicas,
            "scale_down_occurred": scale_down_occurred,
            "replica_history": replica_history,
            "scaling_pattern": "active_scale_down" if scale_down_occurred else "stable" if final_replicas == initial_replicas else "unexpected",
            "timestamp": datetime.now().isoformat()
        }

    def print_phase_result(self, result: Dict[str, Any]):
        """Print formatted phase result"""
        status_emoji = {"SUCCESS": "✅", "FAILED": "❌", "ERROR": "💥"}
        emoji = status_emoji.get(result["status"], "❓")

        phase_name = result["phase"].replace("_", " ").title()
        print(f"{emoji} {phase_name}: {result['status']}")

        # Phase-specific details
        if result["phase"] == "initial_state":
            print(f"   📊 Initial Replicas: {result.get('initial_replicas', 'unknown')}")
            if result.get("scale_to_zero_active"):
                print(f"   💰 Scale-to-zero: ACTIVE (cost = $0/hour)")
            else:
                print(f"   🔄 Scale-to-zero: INACTIVE")

        elif result["phase"] == "cold_start":
            if "request_duration_ms" in result:
                print(f"   ⚡ Cold Start Time: {result['request_duration_ms']}ms")
            if result.get("scaling_occurred"):
                print(f"   📈 Scaling: {result['pre_request_replicas']}→{result['stabilized_replicas']} replicas")
            else:
                print(f"   📊 Replicas: {result.get('stabilized_replicas', 'unknown')} (no change)")

        elif result["phase"] == "concurrent_load":
            if "successful_requests" in result:
                print(f"   📡 Requests: {result['successful_requests']}/{result['total_requests']} successful")
            if result.get("scaling_occurred"):
                print(f"   🚀 Load Scaling: {result['pre_load_replicas']}→{result['post_load_replicas']} replicas")
            if result.get("max_replicas_reached"):
                print(f"   🎯 Maximum replicas (3) reached!")

        elif result["phase"] == "scale_down_monitoring":
            if result.get("scale_down_occurred"):
                print(f"   📉 Scale Down: {result['initial_replicas']}→{result['final_replicas']} replicas")
            else:
                print(f"   📊 Stable at {result['final_replicas']} replicas")

        if result["status"] == "ERROR":
            print(f"   💥 Error: {result.get('error', 'Unknown error')}")

    def print_scaling_summary(self, results: Dict[str, Any]):
        """Print comprehensive scaling demo summary"""
        phases = results["phases"]
        successful_phases = len([p for p in phases if p["status"] == "SUCCESS"])

        print(f"📊 Total Phases: {len(phases)}")
        print(f"✅ Successful: {successful_phases}")
        print(f"📈 Success Rate: {(successful_phases/len(phases))*100:.1f}%")
        print()

        # Scaling events summary
        if self.scaling_events:
            print(f"🔄 Scaling Events Detected:")
            for event in self.scaling_events:
                print(f"   • {event}")
        else:
            print(f"🔄 No scaling events detected")
        print()

        # Auto-scaling assessment
        scale_up_detected = any("→" in event and int(event.split("→")[1].split()[0]) > int(event.split("→")[0].split()[-1]) for event in self.scaling_events)
        scale_down_detected = any("→" in event and int(event.split("→")[1].split()[0]) < int(event.split("→")[0].split()[-1]) for event in self.scaling_events)

        print("🎯 Auto-Scaling Assessment:")
        print(f"   • Scale-Up Capability: {'✅ VERIFIED' if scale_up_detected else '⚠️ NOT OBSERVED'}")
        print(f"   • Scale-Down Capability: {'✅ VERIFIED' if scale_down_detected else '⚠️ NOT OBSERVED'}")
        print(f"   • Cost Optimization: {'✅ ACTIVE' if scale_down_detected or any(p.get('scale_to_zero_active') for p in phases) else '⚠️ MONITORING NEEDED'}")

async def run_demo():
    """Main demo runner"""
    demo = ContainerAppsAutoScalingDemo()
    results = await demo.run_scaling_demo()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"autoscaling_demo_results_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {filename}")
    print(f"🎯 Container Apps URL: {demo.container_url}")
    print("\n🎉 Auto-Scaling Demo Complete!")

if __name__ == "__main__":
    asyncio.run(run_demo())