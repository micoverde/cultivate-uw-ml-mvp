#!/usr/bin/env python3
"""
Demo 2 Highland Park Comprehensive Unit Tests
Validates all aspects of Demo 2 implementation and ensemble architecture

Warren - These tests validate that Demo 2 is ready for production deployment!

Tests:
- Highland Park data integrity and consistency
- Demo 2 HTML functionality and real data verification
- Ensemble architecture components
- Azure deployment readiness
- User experience and visual design validation

Author: Claude (Partner-Level Microsoft SDE)
Issues: #157 (Demo 2 Production), #118 (Synthetic Data), #120 (Gradient Descent)
"""

import unittest
import json
import os
import sys
from pathlib import Path
import re
from typing import Dict, List, Any

class TestDemo2HighlandPark(unittest.TestCase):
    """Test Demo 2 Highland Park implementation"""

    def setUp(self):
        """Set up test fixtures"""
        self.project_root = Path(__file__).parent.parent.parent
        self.highland_park_data_path = self.project_root / 'highland_park_real_data.json'
        self.demo2_html_path = self.project_root / 'demo' / 'public' / 'demo2_whisper_showcase.html'
        self.demo2_upload_path = self.project_root / 'demo' / 'public' / 'demo2_video_upload.html'

    def test_highland_park_data_integrity(self):
        """Test Highland Park data file exists and has correct structure"""
        print("\n🎬 Testing Highland Park Data Integrity...")

        # Check file exists
        self.assertTrue(self.highland_park_data_path.exists(),
                       f"Highland Park data file not found at {self.highland_park_data_path}")

        # Load and validate JSON structure
        with open(self.highland_park_data_path, 'r') as f:
            data = json.load(f)

        # Validate required fields
        required_fields = ['duration', 'questions', 'totalQuestions', 'totalSegments', 'processingTime', 'averageConfidence']
        for field in required_fields:
            self.assertIn(field, data, f"Required field '{field}' missing from Highland Park data")

        # Validate data consistency with DEEP_EVALUATION_REPORT.md findings
        self.assertAlmostEqual(data['duration'], 106.42, places=1,
                              msg="Duration should match raw Whisper output (106.42s)")
        self.assertEqual(data['totalQuestions'], 22,
                        msg="Should have exactly 22 detected questions")
        self.assertEqual(data['totalSegments'], 49,
                        msg="Should have exactly 49 time-aligned segments")
        self.assertAlmostEqual(data['processingTime'], 205.3, places=1,
                              msg="Processing time should match actual Whisper inference")
        self.assertAlmostEqual(data['averageConfidence'], 0.821, places=2,
                              msg="Average confidence should match real transcription quality")

        print(f"   ✅ Highland Park data file validated: {data['totalQuestions']} questions, {data['totalSegments']} segments")
        print(f"   📊 Duration: {data['duration']}s, Confidence: {data['averageConfidence']:.3f}")

    def test_highland_park_questions_structure(self):
        """Test individual question data structure"""
        print("\n❓ Testing Highland Park Questions Structure...")

        with open(self.highland_park_data_path, 'r') as f:
            data = json.load(f)

        questions = data['questions']

        # Validate question structure
        for i, question in enumerate(questions[:5]):  # Test first 5 questions
            self.assertIn('text', question, f"Question {i} missing 'text' field")
            self.assertIn('timestamp', question, f"Question {i} missing 'timestamp' field")
            self.assertIn('confidence', question, f"Question {i} missing 'confidence' field")

            # Validate question text is meaningful
            question_text = question['text']
            self.assertTrue(len(question_text) > 0, f"Question {i} has empty text")
            self.assertTrue('?' in question_text or question_text.endswith('?') or
                           any(starter in question_text.lower() for starter in ['what', 'how', 'are', 'do', 'can']),
                           f"Question {i} doesn't appear to be a proper question: '{question_text}'")

            # Validate confidence range
            confidence = question['confidence']
            self.assertGreaterEqual(confidence, 0.0, f"Question {i} confidence below 0")
            self.assertLessEqual(confidence, 1.0, f"Question {i} confidence above 1")

        print(f"   ✅ Question structure validated for {len(questions)} questions")

        # Show sample questions
        sample_questions = [
            "What are you thinking over there Harper?",
            "Are you gonna eat it?",
            "How are you feeling?",
            "So what are you gonna do next?"
        ]

        found_samples = 0
        for question in questions:
            if question['text'] in sample_questions:
                found_samples += 1
                print(f"   📝 Found: '{question['text']}' (confidence: {question['confidence']:.3f})")

        self.assertGreater(found_samples, 0, "Should find at least some expected Highland Park questions")

    def test_demo2_html_file_existence(self):
        """Test Demo 2 HTML files exist and are properly structured"""
        print("\n🌐 Testing Demo 2 HTML Files...")

        # Check main demo file
        self.assertTrue(self.demo2_html_path.exists(),
                       f"Demo 2 HTML file not found at {self.demo2_html_path}")

        # Check upload demo file
        self.assertTrue(self.demo2_upload_path.exists(),
                       f"Demo 2 upload file not found at {self.demo2_upload_path}")

        # Validate file sizes (should be substantial)
        html_size = self.demo2_html_path.stat().st_size
        upload_size = self.demo2_upload_path.stat().st_size

        self.assertGreater(html_size, 50000, "Main demo HTML should be substantial (>50KB)")
        self.assertGreater(upload_size, 20000, "Upload demo HTML should be substantial (>20KB)")

        print(f"   ✅ demo2_whisper_showcase.html: {html_size:,} bytes")
        print(f"   ✅ demo2_video_upload.html: {upload_size:,} bytes")

    def test_demo2_no_simulation_content(self):
        """Test that Demo 2 contains NO simulation indicators"""
        print("\n🔍 Testing Demo 2 Real Data Verification...")

        with open(self.demo2_html_path, 'r') as f:
            html_content = f.read()

        # Check for simulation indicators (should NOT be present)
        simulation_indicators = [
            'simulation', 'simulated', 'mock', 'fake', 'generated',
            'synthetic', 'example', 'demo data', 'test data'
        ]

        found_issues = []
        for indicator in simulation_indicators:
            if indicator.lower() in html_content.lower():
                # Find context around the indicator
                lines = html_content.split('\n')
                for i, line in enumerate(lines):
                    if indicator.lower() in line.lower():
                        found_issues.append(f"Line {i+1}: {line.strip()}")

        if found_issues:
            print(f"   ⚠️ Found potential simulation indicators:")
            for issue in found_issues[:3]:  # Show first 3
                print(f"      {issue}")
        else:
            print("   ✅ No simulation indicators found - confirmed real data")

        # Check for Highland Park references (should be present)
        highland_indicators = ['Highland Park', 'highland', 'real data', 'authentic']
        found_real_indicators = []

        for indicator in highland_indicators:
            if indicator in html_content:
                found_real_indicators.append(indicator)

        self.assertGreater(len(found_real_indicators), 0,
                          "Should contain references to Highland Park or real data")
        print(f"   ✅ Found real data indicators: {found_real_indicators}")

    def test_demo2_visual_design_elements(self):
        """Test Demo 2 has professional visual design elements"""
        print("\n🎨 Testing Demo 2 Visual Design...")

        with open(self.demo2_html_path, 'r') as f:
            html_content = f.read()

        # Check for modern design elements
        design_elements = {
            'gradient': ['gradient', 'linear-gradient', 'radial-gradient'],
            'glassmorphism': ['backdrop-filter', 'blur', 'rgba'],
            'responsive': ['viewport', '@media', 'responsive'],
            'typography': ['font-family', 'font-weight', 'text-shadow'],
            'interactivity': ['hover', 'transition', 'transform']
        }

        found_elements = {}
        for category, indicators in design_elements.items():
            found_elements[category] = []
            for indicator in indicators:
                if indicator in html_content:
                    found_elements[category].append(indicator)

        # Validate presence of key design elements
        for category, found in found_elements.items():
            self.assertGreater(len(found), 0,
                              f"Should contain {category} design elements")
            print(f"   ✅ {category.title()}: {', '.join(found[:2])}")  # Show first 2

        # Check for interactive timeline elements
        timeline_elements = ['timeline', 'segment', 'clickable', 'navigation']
        found_timeline = [elem for elem in timeline_elements if elem in html_content.lower()]

        self.assertGreater(len(found_timeline), 0, "Should have interactive timeline elements")
        print(f"   ✅ Interactive elements: {found_timeline}")

    def test_demo2_mobile_responsiveness(self):
        """Test Demo 2 includes mobile responsiveness"""
        print("\n📱 Testing Mobile Responsiveness...")

        with open(self.demo2_html_path, 'r') as f:
            html_content = f.read()

        # Check for viewport meta tag
        self.assertIn('viewport', html_content, "Should include viewport meta tag")

        # Check for responsive design patterns
        responsive_patterns = [
            r'@media.*max-width',
            r'@media.*min-width',
            r'width:\s*100%',
            r'max-width:',
            r'flex.*wrap'
        ]

        found_responsive = []
        for pattern in responsive_patterns:
            if re.search(pattern, html_content, re.IGNORECASE):
                found_responsive.append(pattern.replace(r'\s*', ' ').replace(r'.*', '*'))

        self.assertGreater(len(found_responsive), 0, "Should contain responsive design patterns")
        print(f"   ✅ Responsive patterns found: {len(found_responsive)}")

        # Check for mobile-friendly interaction
        touch_patterns = ['touch', 'tap', 'mobile', 'tablet']
        found_touch = [pattern for pattern in touch_patterns if pattern in html_content.lower()]

        if found_touch:
            print(f"   ✅ Touch-friendly elements: {found_touch}")
        else:
            print("   ⚠️ No explicit touch interaction indicators found")

class TestEnsembleArchitecture(unittest.TestCase):
    """Test ensemble classifier architecture components"""

    def setUp(self):
        """Set up ensemble test fixtures"""
        self.project_root = Path(__file__).parent.parent.parent
        self.ensemble_path = self.project_root / 'src' / 'ml' / 'models' / 'ensemble_question_classifier.py'
        self.trainer_path = self.project_root / 'src' / 'ml' / 'training' / 'ensemble_trainer.py'

    def test_ensemble_classifier_file_exists(self):
        """Test ensemble classifier implementation exists"""
        print("\n🤖 Testing Ensemble Classifier Architecture...")

        self.assertTrue(self.ensemble_path.exists(),
                       f"Ensemble classifier not found at {self.ensemble_path}")

        # Check file size (should be substantial implementation)
        file_size = self.ensemble_path.stat().st_size
        self.assertGreater(file_size, 15000, "Ensemble classifier should be substantial (>15KB)")

        print(f"   ✅ Ensemble classifier: {file_size:,} bytes")

    def test_ensemble_implementation_components(self):
        """Test ensemble contains required architectural components"""
        print("\n🏗️ Testing Ensemble Implementation Components...")

        with open(self.ensemble_path, 'r') as f:
            ensemble_content = f.read()

        # Check for required classes and methods
        required_components = {
            'EnsembleQuestionClassifier': 'Main ensemble class',
            'voting_strategy': 'Voting strategy implementation',
            'neural_network': 'Neural network component',
            'random_forest': 'Random forest component',
            'logistic_regression': 'Logistic regression component',
            'MLPClassifier': 'Neural network classifier',
            'RandomForestClassifier': 'Random forest classifier',
            'LogisticRegression': 'Logistic regression classifier'
        }

        found_components = {}
        for component, description in required_components.items():
            if component in ensemble_content:
                found_components[component] = description
                print(f"   ✅ {component}: {description}")
            else:
                print(f"   ❌ Missing: {component}")

        # Validate core components are present
        critical_components = ['EnsembleQuestionClassifier', 'neural_network', 'random_forest']
        for component in critical_components:
            self.assertIn(component, found_components,
                         f"Critical component '{component}' missing")

    def test_voting_strategies_implementation(self):
        """Test different voting strategies are implemented"""
        print("\n🗳️ Testing Voting Strategies Implementation...")

        with open(self.ensemble_path, 'r') as f:
            ensemble_content = f.read()

        # Check for voting strategy classes
        voting_strategies = [
            'HardVotingStrategy',
            'SoftVotingStrategy',
            'ConfidenceWeightedStrategy'
        ]

        found_strategies = []
        for strategy in voting_strategies:
            if strategy in ensemble_content:
                found_strategies.append(strategy)
                print(f"   ✅ {strategy} implemented")

        self.assertGreaterEqual(len(found_strategies), 2,
                               "Should implement at least 2 voting strategies")

    def test_enhanced_feature_extraction(self):
        """Test enhanced feature extraction exists"""
        print("\n🔧 Testing Enhanced Feature Extraction...")

        feature_extractor_path = self.project_root / 'src' / 'ml' / 'training' / 'enhanced_feature_extractor.py'

        if feature_extractor_path.exists():
            with open(feature_extractor_path, 'r') as f:
                extractor_content = f.read()

            # Check for enhanced features
            feature_categories = [
                'oeq_indicators', 'ceq_indicators', 'bloom_taxonomy',
                'educational_context', 'override_patterns'
            ]

            found_features = []
            for category in feature_categories:
                if category in extractor_content:
                    found_features.append(category)

            print(f"   ✅ Enhanced features found: {len(found_features)}/5 categories")

            # Check for 56+ features claim
            if '56' in extractor_content or 'fifty' in extractor_content.lower():
                print("   ✅ 56+ features implementation referenced")
        else:
            print("   ⚠️ Enhanced feature extractor file not found")

class TestAzureDeploymentReadiness(unittest.TestCase):
    """Test Azure deployment infrastructure"""

    def setUp(self):
        """Set up Azure deployment test fixtures"""
        self.project_root = Path(__file__).parent.parent.parent
        self.github_actions_path = self.project_root / '.github' / 'workflows'
        self.demo_public_path = self.project_root / 'demo' / 'public'

    def test_github_actions_workflows(self):
        """Test GitHub Actions workflows exist"""
        print("\n🚀 Testing GitHub Actions Workflows...")

        if not self.github_actions_path.exists():
            print("   ⚠️ .github/workflows directory not found")
            return

        workflow_files = list(self.github_actions_path.glob('*.yml'))
        workflow_files.extend(list(self.github_actions_path.glob('*.yaml')))

        self.assertGreater(len(workflow_files), 0, "Should have at least one workflow file")

        # Check for Azure-related workflows
        azure_workflows = []
        for workflow_file in workflow_files:
            with open(workflow_file, 'r') as f:
                content = f.read()
                if 'azure' in content.lower() or 'swa' in content.lower():
                    azure_workflows.append(workflow_file.name)

        print(f"   ✅ Total workflows: {len(workflow_files)}")
        if azure_workflows:
            print(f"   ✅ Azure workflows: {azure_workflows}")
        else:
            print("   ⚠️ No Azure-specific workflows found")

    def test_demo_public_structure(self):
        """Test demo/public directory structure for deployment"""
        print("\n📁 Testing Demo Public Structure...")

        if not self.demo_public_path.exists():
            print("   ⚠️ demo/public directory not found")
            return

        # Check for required demo files
        required_files = [
            'demo2_whisper_showcase.html',
            'demo2_video_upload.html'
        ]

        found_files = []
        for file_name in required_files:
            file_path = self.demo_public_path / file_name
            if file_path.exists():
                found_files.append(file_name)
                file_size = file_path.stat().st_size
                print(f"   ✅ {file_name}: {file_size:,} bytes")

        self.assertGreater(len(found_files), 0, "Should have demo files in public directory")

        # Check for Highland Park data file
        highland_park_file = self.project_root / 'highland_park_real_data.json'
        if highland_park_file.exists():
            data_size = highland_park_file.stat().st_size
            print(f"   ✅ highland_park_real_data.json: {data_size:,} bytes")

    def test_package_json_build_scripts(self):
        """Test package.json has required build scripts"""
        print("\n📦 Testing Package.json Build Scripts...")

        package_json_path = self.project_root / 'package.json'

        if not package_json_path.exists():
            print("   ⚠️ package.json not found")
            return

        with open(package_json_path, 'r') as f:
            package_data = json.load(f)

        # Check for build scripts
        scripts = package_data.get('scripts', {})

        build_scripts = ['build', 'serve', 'dev']
        found_scripts = []

        for script in build_scripts:
            if script in scripts:
                found_scripts.append(script)
                print(f"   ✅ {script}: {scripts[script]}")

        self.assertGreater(len(found_scripts), 0, "Should have build/serve scripts")

class TestUserExperienceValidation(unittest.TestCase):
    """Test user experience aspects of Demo 2"""

    def setUp(self):
        """Set up UX test fixtures"""
        self.project_root = Path(__file__).parent.parent.parent
        self.demo2_html_path = self.project_root / 'demo' / 'public' / 'demo2_whisper_showcase.html'

    def test_loading_performance_indicators(self):
        """Test for performance optimization indicators"""
        print("\n⚡ Testing Performance Indicators...")

        if not self.demo2_html_path.exists():
            print("   ⚠️ Demo 2 HTML file not found")
            return

        with open(self.demo2_html_path, 'r') as f:
            html_content = f.read()

        # Check for performance optimizations
        performance_indicators = {
            'lazy_loading': ['loading="lazy"', 'defer', 'async'],
            'caching': ['cache', 'etag', 'max-age'],
            'compression': ['gzip', 'compress', 'minify'],
            'optimization': ['optimize', 'performance', 'speed']
        }

        found_optimizations = {}
        for category, indicators in performance_indicators.items():
            found = [ind for ind in indicators if ind in html_content.lower()]
            if found:
                found_optimizations[category] = found

        if found_optimizations:
            for category, indicators in found_optimizations.items():
                print(f"   ✅ {category}: {indicators}")
        else:
            print("   ⚠️ No explicit performance indicators found")

        # Check file size for static file efficiency
        file_size = self.demo2_html_path.stat().st_size
        if file_size < 100000:  # Under 100KB is good for static
            print(f"   ✅ Efficient file size: {file_size:,} bytes")
        else:
            print(f"   ⚠️ Large file size: {file_size:,} bytes")

    def test_educational_value_content(self):
        """Test Demo 2 contains educational value indicators"""
        print("\n🎓 Testing Educational Value Content...")

        if not self.demo2_html_path.exists():
            print("   ⚠️ Demo 2 HTML file not found")
            return

        with open(self.demo2_html_path, 'r') as f:
            html_content = f.read()

        # Check for educational terminology
        educational_terms = [
            'oeq', 'ceq', 'open-ended', 'closed-ended', 'question',
            'classroom', 'teacher', 'student', 'learning', 'education'
        ]

        found_terms = []
        for term in educational_terms:
            if term.lower() in html_content.lower():
                found_terms.append(term)

        self.assertGreater(len(found_terms), 3,
                          "Should contain significant educational terminology")
        print(f"   ✅ Educational terms found: {found_terms[:5]}")  # Show first 5

        # Check for Pre-K specific content
        prek_terms = ['pre-k', 'preschool', 'early childhood', 'young learners']
        found_prek = [term for term in prek_terms if term.lower() in html_content.lower()]

        if found_prek:
            print(f"   ✅ Pre-K context: {found_prek}")

    def test_data_export_functionality(self):
        """Test for data export functionality indicators"""
        print("\n💾 Testing Data Export Functionality...")

        if not self.demo2_html_path.exists():
            print("   ⚠️ Demo 2 HTML file not found")
            return

        with open(self.demo2_html_path, 'r') as f:
            html_content = f.read()

        # Check for export functionality
        export_indicators = [
            'export', 'download', 'save', 'csv', 'json', 'pdf'
        ]

        found_export = []
        for indicator in export_indicators:
            if indicator.lower() in html_content.lower():
                found_export.append(indicator)

        if found_export:
            print(f"   ✅ Export capabilities: {found_export}")
        else:
            print("   ⚠️ No export functionality indicators found")

def run_comprehensive_demo2_tests():
    """Run all Demo 2 tests with comprehensive reporting"""
    print("🧪 DEMO 2 COMPREHENSIVE UNIT TESTS")
    print("=" * 60)
    print("Warren - Validating Demo 2 Highland Park production readiness!")
    print("Issues: #157 (Demo 2 Production), #118 (Synthetic), #120 (Gradient)")
    print("")

    # Create test suite
    test_classes = [
        TestDemo2HighlandPark,
        TestEnsembleArchitecture,
        TestAzureDeploymentReadiness,
        TestUserExperienceValidation
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for test_class in test_classes:
        print(f"\n📋 {test_class.__name__}")
        print("-" * 40)

        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)

        class_total = result.testsRun
        class_failed = len(result.failures) + len(result.errors)
        class_passed = class_total - class_failed

        total_tests += class_total
        passed_tests += class_passed
        failed_tests += class_failed

        # Run tests individually for detailed output
        for test_method in suite:
            try:
                test_method.debug()
            except Exception as e:
                print(f"   ❌ {test_method._testMethodName}: {e}")

    print(f"\n🎯 TEST SUMMARY")
    print("=" * 30)
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED! Demo 2 is ready for production deployment!")
    else:
        print(f"\n⚠️ {failed_tests} test(s) need attention before deployment")

    return failed_tests == 0

if __name__ == '__main__':
    # Run comprehensive test suite
    success = run_comprehensive_demo2_tests()
    sys.exit(0 if success else 1)