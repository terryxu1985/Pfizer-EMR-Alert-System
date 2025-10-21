"""
API Endpoint Test Suite
Tests API endpoint functionality

Uses standard JSON input and output
- Input directory: tests/input/
- Output directory: tests/outputs/
"""
import sys
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


# Test expectations configuration
# Maps test file names to their expected outcomes
TEST_EXPECTATIONS = {
    # Negative test cases (should NOT alert)
    "test_high_risk_senior_patient.json": {
        "should_alert": False,
        "reason": "Clinical criteria met but AI probability (0.669) below threshold (0.7)"
    },
    "test_age_under_12_child.json": {
        "should_alert": False,
        "reason": "Age ineligible (< 12 years old)"
    },
    "test_healthy_young_adult.json": {
        "should_alert": False,
        "reason": "No risk factors, not high-risk population"
    },
    "test_late_symptoms_patient.json": {
        "should_alert": False,
        "reason": "Symptoms beyond 5-day window"
    },
    "test_contraindication_patient.json": {
        "should_alert": False,
        "reason": "Has severe contraindication"
    },
    
    # Positive test cases (SHOULD alert)
    "test_elderly_multiple_risk_factors.json": {
        "should_alert": True,
        "reason": "85-year-old with multiple high-risk conditions (diabetes, heart disease, obesity, COPD) and multiple symptoms"
    },
    "test_immunocompromised_patient.json": {
        "should_alert": True,
        "reason": "72-year-old immunocompromised with diabetes, heart disease, and multiple symptoms within 1 day"
    },
    "test_obese_copd_patient.json": {
        "should_alert": True,
        "reason": "65-year-old with obesity, COPD, diabetes, CVD, and respiratory symptoms"
    }
}


class APIEndpointTestRunner:
    """API endpoint test runner"""
    
    def __init__(self, api_url: Optional[str] = None, use_direct_call: bool = True):
        """
        Initialize test runner
        
        Args:
            api_url: API server address (e.g., http://localhost:8000)
            use_direct_call: Whether to use direct call mode (no server required)
        """
        self.api_url = api_url or "http://localhost:8000"
        self.use_direct_call = use_direct_call
        
        # Directory setup
        self.test_dir = Path(__file__).parent
        self.input_dir = self.test_dir / "input"
        self.output_dir = self.test_dir / "outputs"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # If using direct call mode, import necessary modules
        if self.use_direct_call:
            from backend.api.model_manager import ModelManager
            from backend.api.api_models import RawEMRRequest
            self.ModelManager = ModelManager
            self.RawEMRRequest = RawEMRRequest
            self.model_manager = ModelManager()
            print("✅ Using direct call mode (no API server required)")
        else:
            self.model_manager = None
            print(f"✅ Using HTTP API mode (server: {self.api_url})")
    
    def load_test_case(self, test_file: str) -> dict:
        """Load test case from JSON file"""
        file_path = self.input_dir / test_file
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_test_result(self, test_name: str, result: dict):
        """Save test result to JSON file"""
        output_file = self.output_dir / f"{test_name}_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, indent=2, ensure_ascii=False, fp=f, default=str)
        print(f"✅ Result saved: {output_file}")
    
    def call_api_direct(self, test_data: dict) -> dict:
        """Call model manager directly (no server required)"""
        request = self.RawEMRRequest(**test_data['input'])
        features, _ = self.model_manager.process_raw_emr_data(request)
        result = self.model_manager.predict_single(features)
        return result
    
    def call_api_http(self, test_data: dict) -> dict:
        """Call API via HTTP"""
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/predict",
                json=test_data['input'],
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to API server {self.api_url}. Please start server or use direct mode.")
        except requests.exceptions.Timeout:
            raise Exception("API request timeout")
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")
    
    def run_single_test(self, test_name: str, test_file: str, expected_config: Dict) -> dict:
        """Run a single test case"""
        print("\n" + "="*80)
        print(f"Test case: {test_name}")
        print("="*80)
        
        # Load test data
        test_data = self.load_test_case(test_file)
        print(f"Description: {test_data.get('description', 'N/A')}")
        print(f"Expected: {'Should alert' if expected_config['should_alert'] else 'Should not alert'}")
        print(f"Reason: {expected_config['reason']}")
        print(f"Input file: tests/input/{test_file}")
        print("-"*80)
        
        # Call API
        start_time = datetime.now()
        try:
            if self.use_direct_call:
                result = self.call_api_direct(test_data)
            else:
                result = self.call_api_http(test_data)
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
        except Exception as e:
            print(f"❌ API call failed: {str(e)}")
            return {
                "test_name": test_name,
                "test_file": test_file,
                "error": str(e),
                "test_passed": False
            }
        
        # Print results
        print(f"Response time: {response_time:.3f}s")
        print(f"AI prediction: prediction={result['prediction']}, probability={result['probability']:.3f}")
        print(f"Clinical eligibility: meets_criteria={result['clinical_eligibility']['meets_criteria']}")
        
        clinical = result['clinical_eligibility']
        
        # Print details
        age_status = "✅" if clinical.get('patient_age', 0) >= 12 else "❌"
        print(f"  - Age: {clinical.get('patient_age', 'N/A')} years {age_status}")
        
        window_status = "✅" if clinical.get('within_5day_window') else "❌"
        print(f"  - Within 5 days: {clinical.get('within_5day_window')} {window_status}")
        
        if 'symptom_to_diagnosis_days' in clinical:
            print(f"  - Symptom days: {clinical['symptom_to_diagnosis_days']} days")
        
        risk_status = "✅" if clinical.get('is_high_risk') else "❌"
        print(f"  - High risk: {clinical.get('is_high_risk')} {risk_status}")
        
        if clinical.get('risk_factors_found'):
            print(f"  - Risk factors: {clinical['risk_factors_found']}")
        
        contra_status = "✅" if clinical.get('no_severe_contraindication') else "❌"
        print(f"  - No severe contraindication: {clinical.get('no_severe_contraindication')} {contra_status}")
        
        if 'contraindication_level' in clinical:
            print(f"  - Contraindication level: {clinical['contraindication_level']}")
        
        print(f"Alert recommended: {result['alert_recommended']}")
        
        # Validate result
        alert_recommended = result.get('alert_recommended', False)
        expected_alert = expected_config['should_alert']
        test_passed = (alert_recommended == expected_alert)
        
        # Build test result
        test_result = {
            "test_name": test_name,
            "test_file": test_file,
            "description": test_data.get('description', ''),
            "expected_alert": expected_alert,
            "expected_reason": expected_config['reason'],
            "timestamp": datetime.now().isoformat(),
            "response_time_seconds": response_time,
            "api_mode": "direct_call" if self.use_direct_call else "http",
            "input": test_data['input'],
            "output": result,
            "test_passed": test_passed
        }
        
        # Save result
        self.save_test_result(test_name, test_result)
        
        return test_result
    
    def run_all_tests(self):
        """Run all tests"""
        print("="*80)
        print("API Endpoint Test Suite")
        print("="*80)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"API mode: {'Direct call' if self.use_direct_call else f'HTTP ({self.api_url})'}")
        print("="*80)
        
        # Auto-scan all test files in input directory
        test_files = sorted(self.input_dir.glob("test_*.json"))
        
        if not test_files:
            print("❌ No test files found in tests/input/ directory")
            print("Hint: Test files should start with 'test_' and end with '.json'")
            return
        
        print(f"\nFound {len(test_files)} test file(s):\n")
        for f in test_files:
            # Check if test has expectations defined
            if f.name in TEST_EXPECTATIONS:
                expected = TEST_EXPECTATIONS[f.name]
                alert_str = "✅ Should alert" if expected['should_alert'] else "❌ Should not alert"
                print(f"  - {f.name:<40} {alert_str}")
            else:
                print(f"  - {f.name:<40} ⚠️  No expectations defined")
        
        results = []
        passed = 0
        failed = 0
        errors = 0
        skipped = 0
        
        for test_file_path in test_files:
            test_file = test_file_path.name
            
            # Check if test expectations are defined
            if test_file not in TEST_EXPECTATIONS:
                print(f"\n⚠️  Skipping {test_file} - no expectations defined")
                skipped += 1
                continue
            
            # Extract test name from file name
            test_name = test_file.replace("test_", "").replace(".json", "")
            expected_config = TEST_EXPECTATIONS[test_file]
            
            try:
                result = self.run_single_test(test_name, test_file, expected_config)
                results.append(result)
                
                if 'error' in result:
                    errors += 1
                    print("❌ Test error")
                elif result.get('test_passed') == True:
                    passed += 1
                    print("✅ Test passed")
                elif result.get('test_passed') == False:
                    failed += 1
                    print("❌ Test failed")
                else:
                    print("⚠️  Cannot verify")
                    
            except Exception as e:
                print(f"❌ Test exception: {str(e)}")
                import traceback
                traceback.print_exc()
                errors += 1
                results.append({
                    "test_name": test_name,
                    "test_file": test_file,
                    "error": str(e),
                    "test_passed": False
                })
        
        # Save summary report
        summary = {
            "total_tests": len(test_files),
            "executed": len(results),
            "skipped": skipped,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
            "api_mode": "direct_call" if self.use_direct_call else "http",
            "api_url": self.api_url if not self.use_direct_call else None,
            "test_results": results
        }
        
        summary_file = self.output_dir / "test_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, indent=2, ensure_ascii=False, fp=f, default=str)
        
        # Print summary
        print("\n" + "="*80)
        print("Test Summary")
        print("="*80)
        print(f"Total test files: {len(test_files)}")
        print(f"Executed: {len(results)}")
        print(f"Skipped: {skipped}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Errors: {errors}")
        print(f"Summary report: {summary_file}")
        print("="*80)
        
        return summary


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='API Endpoint Tests')
    parser.add_argument(
        '--mode',
        choices=['direct', 'http'],
        default='direct',
        help='Test mode: direct=direct call (no server), http=HTTP API call'
    )
    parser.add_argument(
        '--api-url',
        default='http://localhost:8000',
        help='API server address (only used in http mode)'
    )
    
    args = parser.parse_args()
    
    use_direct = (args.mode == 'direct')
    runner = APIEndpointTestRunner(
        api_url=args.api_url,
        use_direct_call=use_direct
    )
    
    summary = runner.run_all_tests()
    
    # Return non-zero exit code if there are failures or errors
    if summary and (summary['failed'] > 0 or summary['errors'] > 0):
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
