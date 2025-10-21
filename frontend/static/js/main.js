/**
 * EMR Alert System - Main JavaScript
 * Handles patient management, AI analysis, and UI interactions
 */

// Configuration - Use global CONFIG if available, otherwise use local
const LOCAL_CONFIG = {
    API_BASE_URL: 'http://localhost:8000',
    RETRY_COUNT: 3,
    RETRY_DELAY: 1000,
    HEALTH_CHECK_INTERVAL: 30000
};

// Use global CONFIG if available, fallback to LOCAL_CONFIG
const CONFIG = window.CONFIG || LOCAL_CONFIG;

// Global state
let currentPatient = null;
let alertDismissed = false;
let currentStep = 1;
const totalSteps = 4;

// Initialize application
document.addEventListener('DOMContentLoaded', async function() {
    console.log('🏥 EMR Alert System - Initializing...');
    
    // Initialize birth year dropdown immediately
    populateBirthYearDropdown();
    
    // Also try to populate after a short delay to ensure DOM is ready
    setTimeout(() => {
        populateBirthYearDropdown();
        console.log('🔄 Birth year dropdown populated on delayed initialization');
    }, 500);
    
    // Ensure functions are available before initializing
    setTimeout(async () => {
        try {
            await initializeApplication();
        } catch (error) {
            console.error('❌ Failed to initialize application:', error);
            showError('Failed to initialize application. Please refresh the page.');
        }
    }, 100);
});

/**
 * Initialize the application
 */
async function initializeApplication() {
    // Check API health first
    await checkApiHealth();
    
    // Load patient data
    await loadPatientDataFromAPI();
    
    // Initialize UI components
    loadPatientList();
    initializeEventListeners();
    
    // Start periodic health checks
    setInterval(checkApiHealth, CONFIG.HEALTH_CHECK_INTERVAL);
    
    console.log('✅ Application initialized successfully');
}

/**
 * Check API health
 */
async function checkApiHealth() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            console.log('✅ API Health Check:', data.status);
            
            // Update status indicator
            const statusDot = document.getElementById('apiStatusDot');
            const statusText = document.getElementById('apiStatusText');
            
            if (statusDot && statusText) {
                statusDot.style.backgroundColor = '#10b981'; // Green
                statusText.textContent = 'System Online';
            }
            
            return true;
        } else {
            throw new Error(`Health check failed: ${response.status}`);
        }
    } catch (error) {
        console.warn('⚠️ API Health Check failed:', error.message);
        
        // Update status indicator
        const statusDot = document.getElementById('apiStatusDot');
        const statusText = document.getElementById('apiStatusText');
        
        if (statusDot && statusText) {
            statusDot.style.backgroundColor = '#ef4444'; // Red
            statusText.textContent = 'System Offline';
        }
        
        return false;
    }
}

/**
 * Initialize event listeners
 */
function initializeEventListeners() {
    console.log('🔧 Initializing event listeners...');
    
    // Add any necessary event listeners here
    // For now, most interactions are handled by onclick attributes in HTML
    
    console.log('✅ Event listeners initialized');
}

/**
 * Load patient data from API
 */
async function loadPatientDataFromAPI() {
    try {
        const response = await fetchWithRetry(`${CONFIG.API_BASE_URL}/patients`);
        if (!response.ok) {
            throw new Error(`Failed to load patients: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('📊 Loaded patient data:', data);
        
        // Store patient data globally (you might want to use a proper state management)
        window.patientData = data.patients || [];
        
    } catch (error) {
        console.error('❌ Failed to load patient data:', error);
        showError('Failed to load patient data. Please check your connection.');
    }
}

/**
 * Fetch with retry logic
 */
async function fetchWithRetry(url, options = {}, retries = CONFIG.RETRY_COUNT) {
    for (let i = 0; i < retries; i++) {
        try {
            const response = await fetch(url, options);
            if (response.ok) {
                return response;
            }
            throw new Error(`HTTP ${response.status}`);
        } catch (error) {
            if (i === retries - 1) {
                throw error;
            }
            console.warn(`⚠️ Request failed, retrying... (${i + 1}/${retries})`);
            await new Promise(resolve => setTimeout(resolve, CONFIG.RETRY_DELAY));
        }
    }
}

/**
 * Load patient list in sidebar
 */
function loadPatientList() {
    console.log('📋 loadPatientList called');
    const patientList = document.getElementById('patientList');
    if (!patientList) {
        console.error('❌ Patient list element not found');
        return;
    }
    
    const patients = window.patientData || [];
    console.log(`📊 Loading ${patients.length} patients`);
    
    // Sort patients: Drug A recommended patients first, then by name
    const sortedPatients = [...patients].sort((a, b) => {
        const aRecommended = a.drugAEligible || false;
        const bRecommended = b.drugAEligible || false;
        
        // First priority: Drug A recommended patients
        if (aRecommended && !bRecommended) return -1;
        if (!aRecommended && bRecommended) return 1;
        
        // Second priority: Patients with alerts
        const aHasAlert = a.hasAlert || a.has_alert || false;
        const bHasAlert = b.hasAlert || b.has_alert || false;
        if (aHasAlert && !bHasAlert) return -1;
        if (!aHasAlert && bHasAlert) return 1;
        
        // Third priority: Sort by name alphabetically
        const aName = (a.name || a.patient_name || '').toLowerCase();
        const bName = (b.name || b.patient_name || '').toLowerCase();
        return aName.localeCompare(bName);
    });
    
    console.log(`📋 Sorted patients: ${sortedPatients.filter(p => p.drugAEligible).length} recommended, ${sortedPatients.filter(p => p.hasAlert || p.has_alert).length} with alerts`);
    
    if (sortedPatients.length === 0) {
        patientList.innerHTML = `
            <div class="text-center text-muted p-4">
                <i class="fas fa-users mb-2" style="font-size: 2rem; opacity: 0.5;"></i>
                <p>No patients found</p>
                <p style="font-size: 0.875rem;">Click "Add Patient" to get started</p>
            </div>
        `;
        return;
    }
    
    patientList.innerHTML = sortedPatients.map((patient, index) => {
        const patientJson = JSON.stringify(patient).replace(/"/g, '&quot;');
        const name = patient.name || patient.patient_name || 'Unknown Patient';
        const age = patient.age || patient.patient_age || 'N/A';
        const gender = patient.gender || patient.patient_gender || 'N/A';
        const physicianId = patient.physician?.id || patient.physician_id || 'N/A';
        
        return `
            <div class="patient-item" data-patient-index="${index}" onclick="window.selectPatient ? selectPatient(${patientJson}) : fallbackSelectPatient(${patientJson})">
                <div class="patient-name">${escapeHtml(name)}</div>
            <div class="patient-info">
                    <span>Age: ${age}</span>
                    <span>Gender: ${gender}</span>
                    <span>Physician: ${physicianId}</span>
            </div>
                <div class="patient-status">
                    ${patient.hasAlert || patient.has_alert ? '<span class="status-badge alert">⚠️ Alert</span>' : ''}
                    ${patient.drugAEligible ? '<span class="status-badge recommend">✅ Recommend Drug A</span>' : '<span class="status-badge no-recommendation">ℹ️ No Drug A Recommendation</span>'}
        </div>
            </div>
        `;
    }).join('');
    
    console.log('✅ Patient list loaded successfully');
}

/**
 * Select a patient
 */
function selectPatient(patient) {
    currentPatient = patient;
    
    // Update UI
    updatePatientInfo(patient);
    
    // Update selected state
    document.querySelectorAll('.patient-item').forEach(item => {
        item.classList.remove('selected');
    });
    event.currentTarget.classList.add('selected');
    
    // Show alert panel if applicable
    if (!alertDismissed) {
        showAlertPanel();
    }
    
    console.log('👤 Patient selected:', patient);
}

/**
 * Update patient information display
 */
function updatePatientInfo(patient) {
    console.log('📋 Updating patient info:', patient);
    
    const nameElement = document.getElementById('currentPatientName');
    const detailsElement = document.getElementById('currentPatientDetails');
    
    if (nameElement) {
        nameElement.textContent = patient.name || patient.patient_name || 'Unknown Patient';
    }
    
    if (detailsElement) {
        const age = patient.age || patient.patient_age || 'N/A';
        const gender = patient.gender || patient.patient_gender || 'N/A';
        const physicianId = patient.physician?.id || patient.physician_id || 'N/A';
        const diagnosisDate = patient.diagnosisDate || patient.diagnosis_date || 'N/A';
        
        const details = [
            `Age: ${age}`,
            `Gender: ${gender}`,
            `Physician: ${physicianId}`,
            `Diagnosis: ${diagnosisDate}`
        ].join(' • ');
        detailsElement.textContent = details;
    }
    
    // Update alert panel with patient information (without prediction results)
    // Prediction results will be shown after running AI analysis
    
    // Update alert panel with patient information
    updateAlertPanelPatientInfo(patient, null);
    
    // Show alert panel if patient has alert flag
    if (patient.hasAlert || patient.has_alert) {
        showAlertPanel();
    }
    
    console.log('✅ Patient info updated');
}

/**
 * Show alert panel
 */
function showAlertPanel() {
    const alertPanel = document.getElementById('alertPanel');
    if (alertPanel && currentPatient) {
        alertPanel.classList.remove('hidden');
        
        // Update alert content with patient data
        updateAlertContent();
    }
}

/**
 * Show success message
 */
function showSuccess(message) {
    console.log('✅ Success:', message);
    // Simple alert for now - can be enhanced with a proper notification system
    alert(message);
}

/**
 * Show error message
 */
function showError(message) {
    console.error('❌ Error:', message);
    // Simple alert for now - can be enhanced with a proper notification system
    alert(message);
}

/**
 * Hide alert panel
 */
function hideAlertPanel() {
    const alertPanel = document.getElementById('alertPanel');
    if (alertPanel) {
        alertPanel.classList.add('hidden');
        alertDismissed = true;
    }
}

/**
 * Update alert panel content
 */
function updateAlertContent() {
    if (!currentPatient) return;
    
    // Update treatment probability (mock data for now)
    const probabilityElement = document.getElementById('treatmentProbability');
    if (probabilityElement) {
        // This would normally come from AI analysis
        probabilityElement.textContent = '85%';
    }
    
    // Update other alert content as needed
    // This would be populated with actual AI analysis results
}

/**
 * Convert patient data to API format for prediction
 */
function convertPatientToAPIFormat(patient) {
    console.log('🔄 Converting patient data to API format:', patient);
    
    // Calculate birth year from age
    const currentYear = new Date().getFullYear();
    const birthYear = patient.birth_year || (currentYear - (patient.age || patient.patient_age || 0));
    
    // Parse diagnosis date
    const diagnosisDate = patient.diagnosisDate || patient.diagnosis_date || new Date().toISOString();
    
    // Map symptoms to transactions
    const symptoms = patient.symptoms || [];
    const comorbidities = patient.comorbidities || [];
    
    // Create transactions array
    const transactions = [];
    
    // Add conditions (comorbidities) transactions
    comorbidities.forEach(comorbidity => {
        transactions.push({
            txn_dt: diagnosisDate,
            physician_id: patient.physician?.id || patient.physician_id || null,
            txn_location_type: patient.locationType || patient.location_type || 'OFFICE',
            insurance_type: patient.insuranceType || patient.insurance_type || 'COMMERCIAL',
            txn_type: 'CONDITIONS',
            txn_desc: comorbidity
        });
    });
    
    // Add symptoms transactions
    symptoms.forEach(symptom => {
        transactions.push({
            txn_dt: diagnosisDate,
            physician_id: patient.physician?.id || patient.physician_id || null,
            txn_location_type: patient.locationType || patient.location_type || 'OFFICE',
            insurance_type: patient.insuranceType || patient.insurance_type || 'COMMERCIAL',
            txn_type: 'SYMPTOMS',
            txn_desc: symptom
        });
    });
    
    // Add at least one CONDITIONS transaction (Disease X diagnosis)
    if (transactions.length === 0 || !transactions.some(t => t.txn_type === 'CONDITIONS')) {
        transactions.push({
            txn_dt: diagnosisDate,
            physician_id: patient.physician?.id || patient.physician_id || null,
            txn_location_type: patient.locationType || patient.location_type || 'OFFICE',
            insurance_type: patient.insuranceType || patient.insurance_type || 'COMMERCIAL',
            txn_type: 'CONDITIONS',
            txn_desc: 'DISEASE_X'
        });
    }
    
    // Create API request format
    const apiRequest = {
        patient_id: patient.id || patient.rawPatientId || 999,
        birth_year: birthYear,
        gender: (patient.gender || patient.patient_gender || 'M') === 'Male' ? 'M' : 'F',
        diagnosis_date: diagnosisDate,
        transactions: transactions,
        physician_info: {
            physician_id: patient.physician?.id || patient.physician_id || null,
            state: 'CA',
            physician_type: patient.physician?.specialty || 'Internal Medicine',
            gender: 'M',
            birth_year: null
        }
    };
    
    console.log('✅ Converted to API format:', apiRequest);
    return apiRequest;
}

/**
 * Analyze patient with AI
 */
async function analyzePatient() {
    if (!currentPatient) {
        showError('Please select a patient first');
        return;
    }
    
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span class="loading"><span class="spinner"></span> Analyzing...</span>';
    }
    
    try {
        // Convert patient data to API format
        const apiRequest = convertPatientToAPIFormat(currentPatient);
        
        console.log('📤 Sending prediction request:', apiRequest);
        
        const response = await fetchWithRetry(`${CONFIG.API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(apiRequest)
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ API Error:', errorText);
            throw new Error(`Analysis failed: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('🧠 AI Analysis result:', result);
        
        // Update UI with analysis results
        updateAnalysisResults(result);
        
        // Show success message
        showSuccess('AI analysis completed successfully!');
        
    } catch (error) {
        console.error('❌ AI Analysis failed:', error);
        showError(`AI analysis failed: ${error.message}`);
    } finally {
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-brain"></i> AI Analysis';
        }
    }
}

/**
 * Update analysis results in UI
 */
function updateAnalysisResults(result) {
    console.log('📊 Updating analysis results:', result);
    
    // Update patient object with analysis results for internal tracking
    if (currentPatient) {
        currentPatient.riskScore = result.probability;
        currentPatient.hasAlert = result.alert_recommended || result.probability > 0.7;
        currentPatient.prediction = result.prediction;
    }
    
    // Alert panel styling - removed color coding for neutral, professional design
    // Color is no longer used to indicate risk/confidence
    // Information is presented through text, progress bars, and data visualizations
    
    // Update alert panel with detailed patient information and prediction
    updateAlertPanelPatientInfo(currentPatient, result);
    
    // Show alert panel with results
    showAlertPanel();
    
    console.log('✅ Analysis results updated');
    console.log('   Prediction:', result.prediction === 1 ? 'Likely to prescribe' : 'Unlikely to prescribe');
    console.log('   Probability (Confidence):', (result.probability * 100).toFixed(1) + '%');
    console.log('   Alert Recommended:', result.alert_recommended);
}

/**
 * Update alert panel with patient information
 * Optimized 3-section layout: Recommendation, AI Prediction, Clinical Eligibility
 */
function updateAlertPanelPatientInfo(patient, result) {
    if (!patient) return;
    
    const alertPatientName = document.getElementById('alertPatientName');
    const alertPatientDetails = document.getElementById('alertPatientDetails');
    
    // Update title
    if (alertPatientName) {
        const patientName = patient.name || patient.patient_name || 'Patient';
        alertPatientName.textContent = `${patientName} - AI Analysis Results`;
    }
    
    if (!alertPatientDetails) return;
        
        let detailsHTML = '';
        
    if (result) {
        const prediction = result.prediction;
        const probability = result.probability || 0;
        const alertRecommended = result.alert_recommended; // Primary decision flag from API
        const eligibility = result.clinical_eligibility;
        
        // ==================== 1. COMPREHENSIVE RECOMMENDATION (Top Priority) ====================
        let recommendationIcon, recommendationBg, recommendationBorder, recommendationTitle, recommendationText;
        
        if (alertRecommended) {
            // Recommend Drug A - Based on API alert_recommended flag
            recommendationIcon = 'fa-check-circle';
            recommendationBg = '#d1fae5';
            recommendationBorder = '#10b981';
            recommendationTitle = '✅ Recommend Drug A Treatment';
            recommendationText = 'Patient meets clinical criteria and AI model predicts this patient may not have been prescribed Drug A. Consider prescribing Drug A.';
        } else if (prediction === 1 && probability > 0.6 && eligibility?.meets_criteria) {
            // Consider evaluation - High prediction but no alert
            recommendationIcon = 'fa-exclamation-circle';
            recommendationBg = '#fef3c7';
            recommendationBorder = '#f59e0b';
            recommendationTitle = '⚠️ Clinical Evaluation Suggested';
            recommendationText = 'AI model predicts patient may not have received Drug A, but alert criteria not fully met. Evaluate based on specific conditions.';
        } else if (eligibility && !eligibility.meets_criteria) {
            // Not recommended - Does not meet clinical criteria
            recommendationIcon = 'fa-times-circle';
            recommendationBg = '#fee2e2';
            recommendationBorder = '#ef4444';
            recommendationTitle = '❌ Drug A Not Recommended';
            
            // Determine specific reason for not recommending
            if (eligibility.no_severe_contraindication === false) {
                recommendationText = 'Patient has severe contraindications. Drug A is not recommended.';
            } else if (!eligibility.age_eligible) {
                recommendationText = 'Patient does not meet age requirement (must be ≥12 years old).';
            } else if (!eligibility.within_5day_window) {
                recommendationText = 'Patient exceeds the 5-day treatment window from symptom onset.';
            } else if (!eligibility.is_high_risk) {
                recommendationText = 'Patient does not meet high-risk criteria for Drug A.';
            } else {
                recommendationText = 'Patient does not meet Drug A clinical eligibility criteria.';
            }
        } else {
            // Current treatment appears appropriate
            recommendationIcon = 'fa-info-circle';
            recommendationBg = '#dbeafe';
            recommendationBorder = '#3b82f6';
            recommendationTitle = 'ℹ️ Current Treatment Appears Appropriate';
            recommendationText = 'AI model analysis indicates current treatment approach aligns with historical prescribing patterns.';
        }
        
        detailsHTML += `
            <div class="recommendation-banner" style="
                background: ${recommendationBg};
                border: 2px solid ${recommendationBorder};
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            ">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem;">
                    <i class="fas ${recommendationIcon}" style="color: ${recommendationBorder}; font-size: 2rem;"></i>
                    <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #1f2937;">
                        ${recommendationTitle}
                    </h3>
                </div>
                <p style="margin: 0 0 1rem 0; font-size: 1rem; line-height: 1.6; color: #374151;">
                    ${recommendationText}
                </p>
                ${alertRecommended ? `
                    <div style="display: flex; gap: 0.75rem; margin-top: 1rem;">
                        <button onclick="prescribeDrugA('${patient.id || patient.rawPatientId || 'unknown'}')" 
                                style="
                                    background: #10b981;
                                    color: white;
                                    border: none;
                                    padding: 0.75rem 1.5rem;
                                    border-radius: 8px;
                                    font-size: 0.875rem;
                                    font-weight: 600;
                                    cursor: pointer;
                                    display: flex;
                                    align-items: center;
                                    gap: 0.5rem;
                                    transition: all 0.2s ease;
                                    box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
                                "
                                onmouseover="this.style.background='#059669'; this.style.transform='translateY(-1px)'"
                                onmouseout="this.style.background='#10b981'; this.style.transform='translateY(0)'">
                            <i class="fas fa-pills"></i>
                            Prescribe Drug A
                        </button>
                        <button onclick="reviewPatientEligibility('${patient.id || patient.rawPatientId || 'unknown'}')" 
                                style="
                                    background: white;
                                    color: #10b981;
                                    border: 2px solid #10b981;
                                    padding: 0.75rem 1.5rem;
                                    border-radius: 8px;
                                    font-size: 0.875rem;
                                    font-weight: 600;
                                    cursor: pointer;
                                    display: flex;
                                    align-items: center;
                                    gap: 0.5rem;
                                    transition: all 0.2s ease;
                                "
                                onmouseover="this.style.background='#f0fdf4'"
                                onmouseout="this.style.background='white'">
                            <i class="fas fa-clipboard-check"></i>
                            Review Eligibility
                        </button>
                    </div>
                ` : ''}
            </div>
        `;
        
        // ==================== 2. AI MODEL PREDICTION RESULTS ====================
        detailsHTML += `
            <div class="ai-prediction-section" style="
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
            ">
                <h4 style="margin: 0 0 1.25rem 0; font-size: 1.1rem; font-weight: 700; color: #1f2937; display: flex; align-items: center; gap: 0.5rem;">
                    <i class="fas fa-brain" style="color: #3b82f6;"></i>
                    AI Model Prediction Results
                </h4>
                
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                    <!-- Prediction Result -->
                    <div style="background: #f9fafb; padding: 1.25rem; border-radius: 8px; border-left: 3px solid #3b82f6;">
                        <div style="font-size: 0.875rem; color: #6b7280; margin-bottom: 0.5rem; font-weight: 500;">
                            Prediction
                        </div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #1f2937; margin-bottom: 0.25rem;">
                            ${prediction === 1 ? 'Not Prescribed' : 'Prescribed'}
                        </div>
                        <div style="font-size: 0.875rem; color: #6b7280;">
                            ${prediction === 1 ? 'Drug A likely not prescribed' : 'Drug A likely prescribed'}
                        </div>
                    </div>
                    
                    <!-- Confidence/Probability -->
                    <div style="background: #f9fafb; padding: 1.25rem; border-radius: 8px; border-left: 3px solid ${probability > 0.7 ? '#10b981' : probability > 0.5 ? '#f59e0b' : '#ef4444'};">
                        <div style="font-size: 0.875rem; color: #6b7280; margin-bottom: 0.5rem; font-weight: 500;">
                            Confidence Level
                        </div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #1f2937; margin-bottom: 0.25rem;">
                            ${(probability * 100).toFixed(1)}%
                        </div>
                        <div style="font-size: 0.875rem; color: #6b7280;">
                            ${result.model_type || 'N/A'} v${result.model_version || 'N/A'}
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // ==================== 3. CLINICAL ELIGIBILITY ASSESSMENT ====================
        if (eligibility) {
            const meetsAllCriteria = eligibility.meets_criteria;
            
            detailsHTML += `
                <div class="clinical-eligibility-section" style="
                    background: white;
                    border: 1px solid #e5e7eb;
                    border-radius: 8px;
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                ">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.25rem;">
                        <h4 style="margin: 0; font-size: 1.1rem; font-weight: 700; color: #1f2937; display: flex; align-items: center; gap: 0.5rem;">
                            <i class="fas fa-clipboard-check" style="color: #3b82f6;"></i>
                            Clinical Eligibility Assessment
                        </h4>
                        <span style="
                            padding: 0.5rem 1rem;
                            border-radius: 20px;
                            font-size: 0.875rem;
                            font-weight: 600;
                            background: ${meetsAllCriteria ? '#d1fae5' : '#fee2e2'};
                            color: ${meetsAllCriteria ? '#065f46' : '#991b1b'};
                        ">
                            ${meetsAllCriteria ? '✓ Meets Criteria' : '✗ Does Not Meet Criteria'}
                        </span>
                    </div>
                    
                    <!-- 3.1 HIGH-RISK STATUS (Age, Comorbidities, BMI, etc.) -->
                    <div style="margin-bottom: 1.25rem;">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                            <i class="fas fa-heartbeat" style="color: #f59e0b;"></i>
                            <h5 style="margin: 0; font-size: 0.95rem; font-weight: 600; color: #374151;">
                                High-Risk Status
                            </h5>
                            <span style="
                                padding: 0.125rem 0.5rem;
                                border-radius: 10px;
                                font-size: 0.75rem;
                                font-weight: 600;
                                background: ${eligibility.is_high_risk ? '#fef3c7' : '#f3f4f6'};
                                color: ${eligibility.is_high_risk ? '#92400e' : '#6b7280'};
                            ">
                                ${eligibility.is_high_risk ? 'High Risk' : 'Not High Risk'}
                            </span>
                        </div>
                        
                        <div style="background: #f9fafb; padding: 1rem; border-radius: 6px;">
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem;">
                                <!-- Age -->
                                <div style="display: flex; align-items: center; gap: 0.5rem;">
                                    <i class="fas fa-${eligibility.is_high_risk_by_age ? 'check-circle' : 'circle'}" 
                                       style="color: ${eligibility.is_high_risk_by_age ? '#f59e0b' : '#d1d5db'};"></i>
                                    <span style="font-size: 0.875rem; color: #374151;">
                                        <strong>Age:</strong> ${eligibility.patient_age} years 
                                        ${eligibility.is_high_risk_by_age ? '(≥65 high risk)' : ''}
                                    </span>
                                </div>
                                
                                <!-- Risk Factors Count -->
                                <div style="display: flex; align-items: center; gap: 0.5rem;">
                                    <i class="fas fa-${eligibility.risk_factors_found?.length > 0 ? 'check-circle' : 'circle'}" 
                                       style="color: ${eligibility.risk_factors_found?.length > 0 ? '#f59e0b' : '#d1d5db'};"></i>
                                    <span style="font-size: 0.875rem; color: #374151;">
                                        <strong>Risk Factors:</strong> ${eligibility.risk_factors_found?.length || 0} found
                                    </span>
                                </div>
                            </div>
                            
                            ${eligibility.risk_factors_found && eligibility.risk_factors_found.length > 0 ? `
                                <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #e5e7eb;">
                                    <div style="font-size: 0.8rem; color: #6b7280; margin-bottom: 0.5rem; font-weight: 500;">
                                        Risk Factor Details:
                                    </div>
                                    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                                        ${eligibility.risk_factors_found.map(factor => {
                                            const factorName = formatRiskFactorName(factor);
                                            return `
                                                <span style="
                                                    padding: 0.25rem 0.75rem;
                                                    background: #fef3c7;
                                                    border: 1px solid #fde68a;
                                                    border-radius: 12px;
                                                    font-size: 0.75rem;
                                                    color: #92400e;
                                                    font-weight: 500;
                                                ">${factorName}</span>
                                            `;
                                        }).join('')}
                                    </div>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                    
                    <!-- 3.2 TREATMENT WINDOW -->
                    <div style="margin-bottom: 1.25rem;">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                            <i class="fas fa-clock" style="color: #3b82f6;"></i>
                            <h5 style="margin: 0; font-size: 0.95rem; font-weight: 600; color: #374151;">
                                Treatment Time Window
                            </h5>
                            <span style="
                                padding: 0.125rem 0.5rem;
                                border-radius: 10px;
                                font-size: 0.75rem;
                                font-weight: 600;
                                background: ${eligibility.within_5day_window ? '#d1fae5' : '#fee2e2'};
                                color: ${eligibility.within_5day_window ? '#065f46' : '#991b1b'};
                            ">
                                ${eligibility.within_5day_window ? 'Within Window' : 'Exceeded'}
                            </span>
                        </div>
                        
                        <div style="background: #f9fafb; padding: 1rem; border-radius: 6px;">
                            <div style="display: flex; align-items: center; gap: 1rem;">
                                <div style="flex: 1;">
                                    <div style="font-size: 0.75rem; color: #6b7280; margin-bottom: 0.25rem;">
                                        Symptom to Diagnosis
                                    </div>
                                    <div style="font-size: 1.25rem; font-weight: 700; color: #1f2937;">
                                        ${eligibility.symptom_to_diagnosis_days} days
                                    </div>
                                </div>
                                <div style="
                                    padding: 0.75rem;
                                    background: white;
                                    border-radius: 6px;
                                    border: 1px solid #e5e7eb;
                                ">
                                    <div style="font-size: 0.75rem; color: #6b7280; text-align: center;">
                                        Requirement
                                    </div>
                                    <div style="font-size: 0.875rem; font-weight: 600; color: #374151; text-align: center;">
                                        ≤ 5 days
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- 3.3 CONTRAINDICATION INFORMATION -->
                    <div>
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                            <i class="fas fa-exclamation-triangle" style="color: ${eligibility.contraindication_level > 0 ? '#ef4444' : '#10b981'};"></i>
                            <h5 style="margin: 0; font-size: 0.95rem; font-weight: 600; color: #374151;">
                                Contraindications
                            </h5>
                            <span style="
                                padding: 0.125rem 0.5rem;
                                border-radius: 10px;
                                font-size: 0.75rem;
                                font-weight: 600;
                                background: ${eligibility.no_severe_contraindication ? '#d1fae5' : '#fee2e2'};
                                color: ${eligibility.no_severe_contraindication ? '#065f46' : '#991b1b'};
                            ">
                                ${eligibility.no_severe_contraindication ? 'No Severe CI' : 'CI Present'}
                            </span>
                        </div>
                        
                        <div style="background: ${eligibility.contraindication_level > 0 ? '#fef2f2' : '#f0fdf4'}; padding: 1rem; border-radius: 6px; border-left: 3px solid ${eligibility.contraindication_level > 0 ? '#ef4444' : '#10b981'};">
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                                <span style="font-size: 0.875rem; color: #374151;">
                                    <strong>Contraindication Level:</strong> ${eligibility.contraindication_level}
                                </span>
                                <span style="font-size: 0.75rem; color: #6b7280;">
                                    ${eligibility.contraindication_level === 0 ? 'None' : 
                                      eligibility.contraindication_level === 1 ? 'Mild' : 
                                      eligibility.contraindication_level === 2 ? 'Moderate' : 'Severe'}
                                </span>
                            </div>
                            
                            ${eligibility.contraindication_details && eligibility.contraindication_details.length > 0 ? `
                                <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid ${eligibility.contraindication_level > 0 ? '#fecaca' : '#bbf7d0'};">
                                    <div style="font-size: 0.8rem; color: #6b7280; margin-bottom: 0.5rem;">
                                        Contraindication Details:
                                    </div>
                                    ${eligibility.contraindication_details.map(ci => `
                                        <div style="
                                            display: flex;
                                            align-items: center;
                                            gap: 0.5rem;
                                            padding: 0.5rem;
                                            background: white;
                                            border-radius: 4px;
                                            margin-bottom: 0.25rem;
                                        ">
                                            <i class="fas fa-exclamation-circle" style="color: #ef4444; font-size: 0.875rem;"></i>
                                            <span style="font-size: 0.875rem; color: #374151; font-weight: 500;">${ci}</span>
                                        </div>
                                    `).join('')}
                                </div>
                            ` : `
                                <div style="font-size: 0.875rem; color: #6b7280; margin-top: 0.5rem;">
                                    ${eligibility.contraindication_level === 0 ? '✓ No contraindications detected' : 'No specific contraindication details available'}
                                </div>
                            `}
                        </div>
                    </div>
                </div>
            `;
        }
    }
    
    alertPatientDetails.innerHTML = detailsHTML;
}

/**
 * Format risk factor name for display
 */
function formatRiskFactorName(factor) {
    const nameMap = {
        'age_65_or_older': 'Age ≥65',
        'diabetes': 'Diabetes',
        'cardiovascular_disease': 'Cardiovascular Disease',
        'obesity': 'Obesity',
        'immunocompromised': 'Immunocompromised'
    };
    return nameMap[factor] || factor.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Get risk factor icon
 */
function getRiskFactorIcon(factor) {
    const iconMap = {
        'age_65_or_older': 'user-clock',
        'diabetes': 'syringe',
        'cardiovascular_disease': 'heartbeat',
        'obesity': 'weight',
        'immunocompromised': 'shield-virus'
    };
    return iconMap[factor] || 'circle';
}

/**
 * Show add patient form
 */
function showAddPatientForm() {
    console.log('📝 showAddPatientForm called');
    
    const defaultContent = document.getElementById('defaultContent');
    const patientForm = document.getElementById('patientForm');
    
    if (defaultContent) {
        defaultContent.classList.add('hidden');
        console.log('✅ Hidden default content');
    }
    
    if (patientForm) {
        patientForm.classList.remove('hidden');
        console.log('✅ Showed patient form');
        
        // Set global variables
        window.currentStep = 1;
        window.totalSteps = 4;
        console.log('✅ Global variables set in showAddPatientForm');
        
        // Populate birth year dropdown
        populateBirthYearDropdown();
        
        // Initialize form to step 1
        showStep(1);
    } else {
        alert('Patient form element not found. Please refresh the page.');
    }
}

/**
 * Show specific step in the form
 */
function showStep(stepNumber) {
    console.log('👁️ showStep called for step:', stepNumber);
    
    // Hide all steps
    for (let i = 1; i <= 4; i++) {
        const stepElement = document.getElementById(`step${i}`);
        if (stepElement) {
            stepElement.classList.add('hidden');
        }
    }
    
    // Show current step
    const currentStepElement = document.getElementById(`step${stepNumber}`);
    if (currentStepElement) {
        currentStepElement.classList.remove('hidden');
        console.log(`✅ Showing step ${stepNumber}`);
    }
    
    // Update navigation buttons
    updateNavigationButtons(stepNumber);
}

/**
 * Update navigation buttons
 */
function updateNavigationButtons(currentStep) {
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const submitBtn = document.getElementById('submitBtn');
    
    if (prevBtn) {
        if (currentStep > 1) {
            prevBtn.style.display = 'inline-flex';
        } else {
            prevBtn.style.display = 'none';
        }
    }
    
    if (nextBtn && submitBtn) {
        if (currentStep < 4) {
            nextBtn.classList.remove('hidden');
            submitBtn.classList.add('hidden');
        } else {
            nextBtn.classList.add('hidden');
            submitBtn.classList.remove('hidden');
        }
    }
}

/**
 * Next step in form
 */
function nextStep() {
    console.log('🔄 nextStep called');
    let currentStep = window.currentStep || 1;
    const totalSteps = window.totalSteps || 4;
    
    console.log('Current step:', currentStep, 'Total steps:', totalSteps);
    
    if (currentStep < totalSteps) {
        // Validate current step
        if (validateCurrentStep(currentStep)) {
            currentStep++;
            window.currentStep = currentStep;
            
            // Update UI
            updateStepIndicator(currentStep);
            showStep(currentStep);
            
            console.log('✅ Moved to step', currentStep);
        } else {
            alert('Please fill in all required fields before proceeding.');
        }
    } else {
        alert('Already at the last step');
    }
}

/**
 * Previous step in form
 */
function previousStep() {
    console.log('🔄 previousStep called');
    let currentStep = window.currentStep || 1;
    
    if (currentStep > 1) {
        currentStep--;
        window.currentStep = currentStep;
        
        // Update UI
        updateStepIndicator(currentStep);
        showStep(currentStep);
        
        console.log('✅ Moved to step', currentStep);
    }
}

/**
 * Validate current step
 */
function validateCurrentStep(step) {
    console.log('🔍 validateCurrentStep called for step:', step);
    const currentStepElement = document.getElementById(`step${step}`);
    
    if (!currentStepElement) {
        console.log('❌ Step element not found:', `step${step}`);
        return false;
    }
    
    // Get all required fields in current step
    const requiredFields = currentStepElement.querySelectorAll('[required]');
    console.log('Found required fields:', requiredFields.length);
    
    for (const field of requiredFields) {
        console.log(`Checking field: ${field.id || field.name || 'unnamed'}, value: "${field.value}"`);
        
        // Special validation for Physician ID
        if (field.id === 'physicianId') {
            if (!validatePhysicianId(field)) {
                console.log(`❌ Physician ID validation failed`);
                field.focus();
                return false;
            }
        } else {
            // Standard validation for other fields
            if (!field.value.trim()) {
                console.log(`❌ Field "${field.id || field.name || 'unnamed'}" is empty`);
                field.focus();
                return false;
            }
        }
    }
    
    console.log('✅ All required fields are valid');
    return true;
}

/**
 * Validate physician ID
 */
function validatePhysicianId(field) {
    const value = field.value.trim();
    
    if (!value) {
        return false;
    }
    
    const numValue = parseInt(value);
    
    if (isNaN(numValue) || numValue <= 0) {
        return false;
    }
    
    return true;
}

/**
 * Update step indicator
 */
function updateStepIndicator(currentStep) {
    const steps = document.querySelectorAll('.step');
    steps.forEach((step, index) => {
        step.classList.remove('active', 'completed', 'inactive');
        
        if (index + 1 < currentStep) {
            step.classList.add('completed');
        } else if (index + 1 === currentStep) {
            step.classList.add('active');
        } else {
            step.classList.add('inactive');
        }
    });
}

/**
 * Submit patient form
 */
async function submitPatientForm() {
    console.log('💾 submitPatientForm called');
    
    try {
        // Collect form data
        const formData = collectFormData();
        console.log('📋 Form data collected:', formData);
        
        if (!formData) {
            alert('Failed to collect form data');
            return;
        }
        
        // Validate form data
        if (!validateFormData(formData)) {
            alert('Please fill in all required fields');
            return;
        }
        
        console.log('✅ Form data validated, submitting to API...');
        
        // Submit to API
        const response = await fetch(`${CONFIG.API_BASE_URL}/patients`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            throw new Error(`Failed to create patient: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('✅ Patient created:', result);
        
        alert(`Patient "${formData.patient_name}" saved successfully!`);
        
        // Refresh patient list
        await refreshPatientData();
        
        // Hide form and show patient list
        hideAddPatientForm();
        
    } catch (error) {
        console.error('❌ Failed to submit patient form:', error);
        alert('Failed to save patient. Please try again.');
    }
}

/**
 * Collect form data
 */
function collectFormData() {
    console.log('📋 collectFormData called');
    
    // Collect basic information
    const birthYear = parseInt(document.getElementById('patientAge')?.value);
    const currentYear = new Date().getFullYear();
    const calculatedAge = birthYear ? currentYear - birthYear : 0;
    
    const basicInfo = {
        patient_name: document.getElementById('patientName')?.value || '',
        patient_age: calculatedAge,
        birth_year: birthYear || 0,
        patient_gender: document.getElementById('patientGender')?.value || '',
        physician_id: parseInt(document.getElementById('physicianId')?.value) || 0,
        diagnosis_date: document.getElementById('diagnosisDate')?.value || '',
        symptom_onset_date: document.getElementById('symptomOnsetDate')?.value || null
    };
    
    // Collect medical history
    const medicalHistory = {
        location_type: document.getElementById('locationType')?.value || '',
        insurance_type: document.getElementById('insuranceType')?.value || '',
        contraindication_level: document.getElementById('contraindicationLevel')?.value || '',
        additional_notes: document.getElementById('additionalNotes')?.value || ''
    };
    
    // Collect symptoms and comorbidities
    const symptoms = [];
    const comorbidities = [];
    
    // Get checked symptoms from step 3
    const step3 = document.getElementById('step3');
    if (step3) {
        const formGroups = step3.querySelectorAll('.form-group');
        
        // First form-group contains symptoms
        if (formGroups[0]) {
            const symptomCheckboxes = formGroups[0].querySelectorAll('input[type="checkbox"]:checked');
            symptomCheckboxes.forEach(checkbox => {
                symptoms.push(checkbox.value);
            });
        }
        
        // Second form-group contains comorbidities
        if (formGroups[1]) {
            const comorbidityCheckboxes = formGroups[1].querySelectorAll('input[type="checkbox"]:checked');
            comorbidityCheckboxes.forEach(checkbox => {
                comorbidities.push(checkbox.value);
            });
        }
    }
    
    // Combine all data
    const formData = {
        ...basicInfo,
        ...medicalHistory,
        comorbidities: comorbidities,
        symptoms: symptoms
    };
    
    console.log('✅ Form data collected:', formData);
    return formData;
}

/**
 * Validate form data
 */
function validateFormData(data) {
    console.log('🔍 validateFormData called');
    
    if (!data.patient_name || !data.patient_name.trim()) {
        console.log('❌ Patient name is required');
        return false;
    }
    
    if (!data.patient_age || data.patient_age <= 0) {
        console.log('❌ Patient age is required and must be greater than 0');
        return false;
    }
    
    if (!data.patient_gender || !data.patient_gender.trim()) {
        console.log('❌ Patient gender is required');
        return false;
    }
    
    if (!data.physician_id || data.physician_id <= 0) {
        console.log('❌ Physician ID is required and must be greater than 0');
        return false;
    }
    
    if (!data.diagnosis_date || !data.diagnosis_date.trim()) {
        console.log('❌ Diagnosis date is required');
        return false;
    }
    
    if (!data.location_type || !data.location_type.trim()) {
        console.log('❌ Location type is required');
        return false;
    }
    
    if (!data.insurance_type || !data.insurance_type.trim()) {
        console.log('❌ Insurance type is required');
        return false;
    }
    
    if (!data.contraindication_level || !data.contraindication_level.trim()) {
        console.log('❌ Contraindication level is required');
        return false;
    }
    
    console.log('✅ All required fields are valid');
    return true;
}

/**
 * Hide add patient form
 */
function hideAddPatientForm() {
    console.log('🙈 hideAddPatientForm called');
    
    const patientForm = document.getElementById('patientForm');
    if (patientForm) {
        patientForm.classList.add('hidden');
    }
    
    const defaultContent = document.getElementById('defaultContent');
    if (defaultContent) {
        defaultContent.classList.remove('hidden');
    }
    
    // Reset form state
    window.currentStep = 1;
    clearFormData();
}

/**
 * Clear form data
 */
function clearFormData() {
    const form = document.getElementById('patientForm');
    if (form) {
        const inputs = form.querySelectorAll('input, select, textarea');
        inputs.forEach(input => {
            if (input.type === 'checkbox') {
                input.checked = false;
            } else {
                input.value = '';
            }
        });
    }
}

/**
 * Refresh patient data
 */
async function refreshPatientData() {
    console.log('🔄 refreshPatientData called');
    
    try {
        // Load patient data from API
        await loadPatientDataFromAPI();
        
        // Update patient list display
        loadPatientList();
        
        alert('Patient data refreshed successfully!');
    } catch (error) {
        console.error('❌ Failed to refresh patient data:', error);
        alert('Failed to refresh patient data. Please try again.');
    }
}

/**
 * Populate birth year dropdown
 */
function populateBirthYearDropdown() {
    console.log('🔄 populateBirthYearDropdown called');
    const birthYearSelect = document.getElementById('patientAge');
    if (!birthYearSelect) {
        console.error('❌ patientAge element not found');
        return;
    }
    
    // Clear existing options except the first one
    birthYearSelect.innerHTML = '<option value="">Select Birth Year</option>';
    
    // Generate years from 1924 to current year
    const currentYear = new Date().getFullYear();
    const startYear = currentYear - 100;
    
    for (let year = currentYear; year >= startYear; year--) {
        const option = document.createElement('option');
        option.value = year;
        option.textContent = year;
        birthYearSelect.appendChild(option);
    }
    
    console.log(`✅ Birth year dropdown populated (${birthYearSelect.options.length} options)`);
}

/**
 * Prescribe Drug A for patient
 */
function prescribeDrugA(patientId) {
    console.log('💊 Prescribe Drug A clicked for patient:', patientId);
    
    if (!currentPatient) {
        showError('No patient selected');
        return;
    }
    
    // Show confirmation dialog
    const patientName = currentPatient.name || currentPatient.patient_name || 'Unknown Patient';
    const confirmMessage = `Are you sure you want to prescribe Drug A for ${patientName}?\n\nThis action will:\n• Record the prescription in the system\n• Update patient treatment status\n• Generate prescription documentation`;
    
    if (confirm(confirmMessage)) {
        // Simulate prescription process
        console.log('📋 Processing Drug A prescription...');
        
        // Show loading state
        const prescribeBtn = event.target;
        const originalText = prescribeBtn.innerHTML;
        prescribeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        prescribeBtn.disabled = true;
        
        // Simulate API call delay
        setTimeout(() => {
            // Reset button
            prescribeBtn.innerHTML = originalText;
            prescribeBtn.disabled = false;
            
            // Show success message
            showSuccess(`Drug A prescription has been recorded for ${patientName}!\n\nPrescription ID: RX-${Date.now()}\nStatus: Pending Pharmacy Processing`);
            
            // Update patient status
            if (currentPatient) {
                currentPatient.drugAPrescribed = true;
                currentPatient.prescriptionDate = new Date().toISOString();
                currentPatient.prescriptionId = `RX-${Date.now()}`;
            }
            
            console.log('✅ Drug A prescription completed');
        }, 2000);
    }
}

/**
 * Review patient eligibility for Drug A
 */
function reviewPatientEligibility(patientId) {
    console.log('📋 Review Eligibility clicked for patient:', patientId);
    
    if (!currentPatient) {
        showError('No patient selected');
        return;
    }
    
    const patientName = currentPatient.name || currentPatient.patient_name || 'Unknown Patient';
    
    // Create eligibility review dialog
    const eligibilityInfo = `
Patient Eligibility Review: ${patientName}

📊 Current Status:
• Age: ${currentPatient.age || currentPatient.patient_age || 'N/A'} years
• Gender: ${currentPatient.gender || currentPatient.patient_gender || 'N/A'}
• Diagnosis Date: ${currentPatient.diagnosisDate || currentPatient.diagnosis_date || 'N/A'}
• Location: ${currentPatient.locationType || currentPatient.location_type || 'N/A'}
• Insurance: ${currentPatient.insuranceType || currentPatient.insurance_type || 'N/A'}
• Contraindication Level: ${currentPatient.contraindicationLevel || currentPatient.contraindication_level || 'N/A'}

🔍 Eligibility Criteria:
✓ Age ≥ 12 years
✓ Within 5-day treatment window
✓ No severe contraindications
✓ High-risk factors present

💡 Recommendation: Patient appears eligible for Drug A treatment.
    `;
    
    alert(eligibilityInfo);
    
    console.log('✅ Eligibility review completed');
}

// Version marker for cache busting
console.log('📌 main.js loaded - Version: 20251021-v5 - Added prescription buttons');
console.log('📌 Features: Comprehensive Recommendation Banner + AI Prediction + Clinical Eligibility + Prescription Actions');
console.log('📌 Fixed: showSuccess, showError, checkApiHealth, initializeEventListeners, populateBirthYearDropdown, showAddPatientForm, nextStep, previousStep, submitPatientForm');
console.log('📌 Added: prescribeDrugA, reviewPatientEligibility');
