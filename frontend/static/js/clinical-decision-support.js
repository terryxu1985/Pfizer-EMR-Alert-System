/**
 * Clinical Decision Support Module
 * 临床决策支持模块 - 风险因素可视化和交互式决策流程
 */

class ClinicalDecisionSupport {
    constructor() {
        this.currentPatient = null;
        this.currentPrediction = null;
    }

    /**
     * 初始化临床决策支持界面
     */
    initialize(patient, prediction) {
        this.currentPatient = patient;
        this.currentPrediction = prediction;
        
        this.renderClinicalEligibilityCard();
        this.renderRiskFactorVisualization();
        this.renderDecisionSupportFlow();
    }

    /**
     * 渲染临床资格评估卡片
     */
    renderClinicalEligibilityCard() {
        const eligibility = this.currentPrediction?.clinical_eligibility;
        if (!eligibility) return;

        const cardHTML = `
            <div class="clinical-eligibility-card" id="clinicalEligibilityCard">
                <div class="card-header">
                    <div class="header-left">
                        <i class="fas fa-clipboard-check"></i>
                        <h3>Eligible for Drug A Treatment</h3>
                    </div>
                    <div class="eligibility-badge ${eligibility.meets_criteria ? 'eligible' : 'not-eligible'}">
                        ${eligibility.meets_criteria ? 
                            '<i class="fas fa-check-circle"></i> Eligible' : 
                            '<i class="fas fa-times-circle"></i> Not Eligible'}
                    </div>
                </div>

                <div class="card-body">
                    <!-- 核心资格标准 -->
                    <div class="criteria-grid">
                        ${this.renderCriteriaItem(
                            'age_eligible', 
                            eligibility.age_eligible,
                            'User',
                            `Age: ${eligibility.patient_age} years`,
                            eligibility.age_eligible ? 
                                `Patient is ${eligibility.patient_age} years old (≥12 required)` :
                                `Patient is ${eligibility.patient_age} years old (must be ≥12)`
                        )}
                        
                        ${this.renderCriteriaItem(
                            'within_5day_window',
                            eligibility.within_5day_window,
                            'Clock',
                            `${eligibility.symptom_to_diagnosis_days} days to diagnosis`,
                            eligibility.within_5day_window ?
                                `Symptoms started ${eligibility.symptom_to_diagnosis_days} days ago (within 5-day window)` :
                                `Symptoms started ${eligibility.symptom_to_diagnosis_days} days ago (exceeds 5-day window)`
                        )}
                        
                        ${this.renderCriteriaItem(
                            'is_high_risk',
                            eligibility.is_high_risk,
                            'Exclamation-Triangle',
                            `${(eligibility.risk_factors_found || []).length} risk factors`,
                            eligibility.is_high_risk ?
                                `High-risk patient identified` :
                                `No high-risk factors identified`
                        )}
                        
                        ${this.renderCriteriaItem(
                            'no_severe_contraindication',
                            eligibility.no_severe_contraindication,
                            'Shield-Alt',
                            `Level ${eligibility.contraindication_level}`,
                            eligibility.no_severe_contraindication ?
                                `No severe contraindications (Level ${eligibility.contraindication_level})` :
                                `Severe contraindication present (Level ${eligibility.contraindication_level})`
                        )}
                    </div>

                    <!-- 风险因素详情 -->
                    ${(eligibility.risk_factors_found && eligibility.risk_factors_found.length > 0) ?
                        this.renderRiskFactorsSection(eligibility) : ''}

                    <!-- 禁忌症详情 -->
                    ${(eligibility.contraindication_details && eligibility.contraindication_details.length > 0) ?
                        this.renderContraindicationsSection(eligibility) : ''}
                </div>
            </div>
        `;

        return cardHTML;
    }

    /**
     * 渲染单个标准项
     */
    renderCriteriaItem(id, passed, icon, label, description) {
        const statusClass = passed ? 'criteria-passed' : 'criteria-failed';
        const statusIcon = passed ? 'fa-check-circle' : 'fa-times-circle';
        const statusColor = passed ? '#10b981' : '#ef4444';

        return `
            <div class="criteria-item ${statusClass}" data-criteria="${id}">
                <div class="criteria-icon">
                    <i class="fas fa-${icon.toLowerCase()}" style="color: ${statusColor};"></i>
                </div>
                <div class="criteria-content">
                    <div class="criteria-label">${label}</div>
                    <div class="criteria-description">${description}</div>
                </div>
                <div class="criteria-status">
                    <i class="fas ${statusIcon}" style="color: ${statusColor};"></i>
                </div>
            </div>
        `;
    }

    /**
     * 渲染风险因素部分
     */
    renderRiskFactorsSection(eligibility) {
        const riskFactors = eligibility.risk_factors_found || [];
        const riskDetails = eligibility.risk_conditions_details || {};

        return `
            <div class="risk-factors-section">
                <div class="section-header">
                    <i class="fas fa-heartbeat"></i>
                    <h4>Identified Risk Factors</h4>
                    <span class="risk-count-badge">${riskFactors.length}</span>
                </div>
                <div class="risk-factors-list">
                    ${riskFactors.map(factor => {
                        const details = riskDetails[factor] || [];
                        return `
                            <div class="risk-factor-item">
                                <div class="risk-factor-name">
                                    <i class="fas fa-circle" style="color: #f59e0b; font-size: 0.5rem;"></i>
                                    ${this.formatRiskFactorName(factor)}
                                </div>
                                ${details.length > 0 ? `
                                    <div class="risk-factor-details">
                                        ${details.map(d => `<span class="detail-tag">${d}</span>`).join('')}
                                    </div>
                                ` : ''}
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
    }

    /**
     * 渲染禁忌症部分
     */
    renderContraindicationsSection(eligibility) {
        const contraindications = eligibility.contraindication_details || [];
        const level = eligibility.contraindication_level;

        let severityClass = '';
        let severityLabel = '';
        if (level === 3) {
            severityClass = 'severe';
            severityLabel = 'Severe';
        } else if (level === 2) {
            severityClass = 'moderate';
            severityLabel = 'Moderate';
        } else if (level === 1) {
            severityClass = 'mild';
            severityLabel = 'Mild';
        } else {
            severityClass = 'none';
            severityLabel = 'None';
        }

        return `
            <div class="contraindications-section ${severityClass}">
                <div class="section-header">
                    <i class="fas fa-exclamation-triangle"></i>
                    <h4>Contraindications</h4>
                    <span class="severity-badge ${severityClass}">${severityLabel} (Level ${level})</span>
                </div>
                ${contraindications.length > 0 ? `
                    <div class="contraindications-list">
                        ${contraindications.map(ci => `
                            <div class="contraindication-item">
                                <i class="fas fa-exclamation-circle"></i>
                                <span>${ci}</span>
                            </div>
                        `).join('')}
                    </div>
                ` : `
                    <div class="no-contraindications">
                        <i class="fas fa-check-circle" style="color: #10b981;"></i>
                        <span>No contraindications identified</span>
                    </div>
                `}
            </div>
        `;
    }

    /**
     * 渲染风险因素可视化 - 仅条形图
     */
    renderRiskFactorVisualization() {
        const eligibility = this.currentPrediction?.clinical_eligibility;
        if (!eligibility) return;

        const riskFactors = eligibility.risk_factors_found || [];
        
        const visualizationHTML = `
            <div class="risk-visualization-container" id="riskVisualizationContainer">
                <div class="visualization-header">
                    <h3>Risk Factor Analysis</h3>
                </div>

                <!-- 条形图视图 -->
                <div class="visualization-view bars-view active" id="barsView">
                    ${this.renderBarChart(riskFactors)}
                </div>
            </div>
        `;

        return visualizationHTML;
    }

    /**
     * 渲染雷达图（CSS版本）
     */
    renderRadarChart(riskFactors) {
        const riskCategories = {
            'age_65_or_older': { label: 'Age', value: riskFactors.includes('age_65_or_older') ? 100 : 0 },
            'cardiovascular_disease': { label: 'CVD', value: riskFactors.includes('cardiovascular_disease') ? 100 : 0 },
            'diabetes': { label: 'Diabetes', value: riskFactors.includes('diabetes') ? 100 : 0 },
            'obesity': { label: 'Obesity', value: riskFactors.includes('obesity') ? 100 : 0 },
            'chronic_lung_disease': { label: 'Lung', value: riskFactors.includes('chronic_lung_disease') ? 100 : 0 },
            'immunocompromised': { label: 'Immune', value: riskFactors.includes('immunocompromised') ? 100 : 0 }
        };

        return `
            <div class="radar-chart">
                <svg viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
                    <!-- 背景网格 -->
                    <g class="radar-grid">
                        ${[100, 80, 60, 40, 20].map(radius => `
                            <circle cx="200" cy="200" r="${radius}" 
                                fill="none" stroke="#e5e7eb" stroke-width="1"/>
                        `).join('')}
                        
                        <!-- 轴线 -->
                        ${Object.keys(riskCategories).map((key, index) => {
                            const angle = (index * 60 - 90) * Math.PI / 180;
                            const x = 200 + 100 * Math.cos(angle);
                            const y = 200 + 100 * Math.sin(angle);
                            return `<line x1="200" y1="200" x2="${x}" y2="${y}" 
                                stroke="#d1d5db" stroke-width="1"/>`;
                        }).join('')}
                    </g>

                    <!-- 数据区域 -->
                    <polygon 
                        points="${Object.keys(riskCategories).map((key, index) => {
                            const risk = riskCategories[key];
                            const angle = (index * 60 - 90) * Math.PI / 180;
                            const distance = risk.value; // 0-100
                            const x = 200 + distance * Math.cos(angle);
                            const y = 200 + distance * Math.sin(angle);
                            return `${x},${y}`;
                        }).join(' ')}"
                        fill="rgba(37, 99, 235, 0.2)"
                        stroke="rgba(37, 99, 235, 0.8)"
                        stroke-width="2"
                    />

                    <!-- 标签 -->
                    ${Object.keys(riskCategories).map((key, index) => {
                        const risk = riskCategories[key];
                        const angle = (index * 60 - 90) * Math.PI / 180;
                        const x = 200 + 120 * Math.cos(angle);
                        const y = 200 + 120 * Math.sin(angle);
                        const active = risk.value > 0;
                        return `
                            <text x="${x}" y="${y}" 
                                text-anchor="middle" 
                                dominant-baseline="middle"
                                fill="${active ? '#2563eb' : '#9ca3af'}"
                                font-weight="${active ? '600' : '400'}"
                                font-size="14">
                                ${risk.label}
                            </text>
                        `;
                    }).join('')}
                </svg>
                
                <div class="radar-legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background: rgba(37, 99, 235, 0.2); border: 2px solid rgba(37, 99, 235, 0.8);"></div>
                        <span>Risk Profile</span>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * 渲染条形图
     */
    renderBarChart(riskFactors) {
        const allRiskTypes = [
            { key: 'age_65_or_older', label: 'Age ≥65 Years', icon: 'user-clock' },
            { key: 'cardiovascular_disease', label: 'Cardiovascular Disease', icon: 'heartbeat' },
            { key: 'diabetes', label: 'Diabetes', icon: 'syringe' },
            { key: 'obesity', label: 'Obesity (BMI ≥30)', icon: 'weight' },
            { key: 'chronic_lung_disease', label: 'Chronic Lung Disease', icon: 'lungs' },
            { key: 'immunocompromised', label: 'Immunocompromised', icon: 'shield-virus' }
        ];

        return `
            <div class="bar-chart">
                ${allRiskTypes.map(risk => {
                    const present = riskFactors.includes(risk.key);
                    return `
                        <div class="bar-item ${present ? 'present' : 'absent'}">
                            <div class="bar-label">
                                <i class="fas fa-${risk.icon}"></i>
                                <span>${risk.label}</span>
                            </div>
                            <div class="bar-container">
                                <div class="bar-fill ${present ? 'filled' : ''}" 
                                    style="width: ${present ? '100%' : '0%'};">
                                </div>
                            </div>
                            <div class="bar-status">
                                ${present ? 
                                    '<i class="fas fa-check-circle" style="color: #f59e0b;"></i>' : 
                                    '<i class="fas fa-minus-circle" style="color: #d1d5db;"></i>'}
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }

    /**
     * 渲染标签云
     */
    renderTagCloud(riskFactors) {
        const tags = riskFactors.map(factor => ({
            text: this.formatRiskFactorName(factor),
            size: Math.floor(Math.random() * 3) + 1 // 1-3
        }));

        return `
            <div class="tag-cloud">
                ${tags.map(tag => `
                    <span class="cloud-tag size-${tag.size}">
                        ${tag.text}
                    </span>
                `).join('')}
                
                ${tags.length === 0 ? `
                    <div class="no-risk-factors">
                        <i class="fas fa-shield-alt" style="font-size: 3rem; color: #10b981; opacity: 0.5;"></i>
                        <p>No significant risk factors identified</p>
                    </div>
                ` : ''}
            </div>
        `;
    }


    /**
     * 渲染医生决策支持流程
     */
    renderDecisionSupportFlow() {
        const prediction = this.currentPrediction;
        const eligibility = prediction?.clinical_eligibility;
        
        if (!prediction || !eligibility) return;

        const flowHTML = `
            <div class="decision-support-flow" id="decisionSupportFlow">
                <div class="flow-header">
                    <i class="fas fa-route"></i>
                    <h3>Clinical Decision Support</h3>
                </div>

                <div class="flow-body">
                    <!-- 决策树 -->
                    ${this.renderDecisionTree(prediction, eligibility)}
                </div>
            </div>
        `;

        return flowHTML;
    }

    /**
     * 渲染决策树
     */
    renderDecisionTree(prediction, eligibility) {
        const steps = [
            {
                id: 'step1',
                title: 'Patient Assessment',
                status: 'completed',
                description: `${eligibility.patient_age} years old, ${eligibility.risk_factors_found?.length || 0} risk factors`,
                icon: 'user-check'
            },
            {
                id: 'step2',
                title: 'Eligibility Check',
                status: eligibility.meets_criteria ? 'completed' : 'warning',
                description: eligibility.meets_criteria ? 'All criteria met' : 'Some criteria not met',
                icon: 'clipboard-check'
            },
            {
                id: 'step3',
                title: 'AI Analysis',
                status: 'completed',
                description: `${(prediction.probability * 100).toFixed(1)}% confidence`,
                icon: 'brain'
            },
            {
                id: 'step4',
                title: 'Recommendation',
                status: prediction.alert_recommended ? 'action-required' : 'completed',
                description: prediction.alert_recommended ? 
                    'Consider Drug A prescription' : 
                    'Continue current treatment',
                icon: 'prescription'
            }
        ];

        return `
            <div class="decision-tree">
                ${steps.map((step, index) => `
                    <div class="tree-step ${step.status}" data-step="${step.id}">
                        <div class="step-icon">
                            <i class="fas fa-${step.icon}"></i>
                        </div>
                        <div class="step-content">
                            <div class="step-title">${step.title}</div>
                            <div class="step-description">${step.description}</div>
                        </div>
                        ${index < steps.length - 1 ? `
                            <div class="step-connector">
                                <i class="fas fa-arrow-down"></i>
                            </div>
                        ` : ''}
                    </div>
                `).join('')}
            </div>
        `;
    }

    /**
     * 渲染推荐行动
     */
    renderRecommendedActions(prediction, eligibility) {
        let actions = [];

        if (prediction.alert_recommended && eligibility.meets_criteria) {
            actions.push({
                priority: 'high',
                icon: 'pills',
                title: 'Consider Drug A Prescription',
                description: 'Patient meets all clinical criteria and AI indicates high probability of benefit',
                steps: [
                    'Review patient contraindications',
                    'Discuss treatment options with patient',
                    'Document decision rationale',
                    'Prescribe Drug A if appropriate'
                ]
            });
        } else if (prediction.prediction === 1 && !prediction.alert_recommended) {
            actions.push({
                priority: 'medium',
                icon: 'stethoscope',
                title: 'Clinical Review Recommended',
                description: 'AI suggests potential benefit but confidence is moderate',
                steps: [
                    'Review patient risk factors',
                    'Assess symptom severity',
                    'Consider alternative treatments',
                    'Schedule follow-up if needed'
                ]
            });
        } else {
            actions.push({
                priority: 'low',
                icon: 'check-circle',
                title: 'Continue Current Treatment',
                description: 'Current treatment approach appears appropriate',
                steps: [
                    'Monitor patient progress',
                    'Schedule regular follow-ups',
                    'Reassess if condition changes'
                ]
            });
        }

        // 总是添加文档建议
        actions.push({
            priority: 'info',
            icon: 'file-medical',
            title: 'Documentation',
            description: 'Document clinical decision and AI analysis results',
            steps: [
                'Record AI prediction and probability',
                'Note risk factors considered',
                'Document clinical reasoning',
                'Update patient treatment plan'
            ]
        });

        return `
            <div class="recommended-actions">
                <h4>Recommended Actions</h4>
                ${actions.map(action => `
                    <div class="action-card priority-${action.priority}">
                        <div class="action-header">
                            <div class="action-icon">
                                <i class="fas fa-${action.icon}"></i>
                            </div>
                            <div class="action-title-group">
                                <h5>${action.title}</h5>
                                <p>${action.description}</p>
                            </div>
                        </div>
                        <div class="action-steps">
                            <ol>
                                ${action.steps.map(step => `<li>${step}</li>`).join('')}
                            </ol>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    /**
     * 渲染证据支持
     */
    renderEvidenceSupport(prediction, eligibility) {
        const confidence = prediction.probability;
        let confidenceLevel = '';
        let confidenceDescription = '';

        if (confidence > 0.8) {
            confidenceLevel = 'High';
            confidenceDescription = 'AI model shows strong confidence based on similar historical cases';
        } else if (confidence > 0.6) {
            confidenceLevel = 'Moderate';
            confidenceDescription = 'AI model shows moderate confidence, clinical judgment important';
        } else {
            confidenceLevel = 'Low';
            confidenceDescription = 'AI model shows low confidence, prioritize clinical assessment';
        }

        return `
            <div class="evidence-support">
                <h4>Evidence & Analysis</h4>
                
                <div class="evidence-grid">
                    <div class="evidence-item">
                        <div class="evidence-icon">
                            <i class="fas fa-brain"></i>
                        </div>
                        <div class="evidence-content">
                            <h5>AI Confidence Level</h5>
                            <div class="confidence-meter">
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: ${confidence * 100}%;"></div>
                                </div>
                                <span class="confidence-value">${confidenceLevel} (${(confidence * 100).toFixed(1)}%)</span>
                            </div>
                            <p>${confidenceDescription}</p>
                        </div>
                    </div>

                    <div class="evidence-item">
                        <div class="evidence-icon">
                            <i class="fas fa-database"></i>
                        </div>
                        <div class="evidence-content">
                            <h5>Model Information</h5>
                            <ul>
                                <li><strong>Version:</strong> ${prediction.model_version || 'N/A'}</li>
                                <li><strong>Type:</strong> ${prediction.model_type || 'N/A'}</li>
                                <li><strong>Processing Time:</strong> ${prediction.processing_time_ms?.toFixed(2) || 'N/A'}ms</li>
                            </ul>
                        </div>
                    </div>

                    <div class="evidence-item">
                        <div class="evidence-icon">
                            <i class="fas fa-chart-line"></i>
                        </div>
                        <div class="evidence-content">
                            <h5>Key Factors Analyzed</h5>
                            <ul>
                                <li>Patient demographics and age</li>
                                <li>${eligibility.risk_factors_found?.length || 0} risk factors identified</li>
                                <li>Symptom onset timing (${eligibility.symptom_to_diagnosis_days} days)</li>
                                <li>Contraindication level (${eligibility.contraindication_level})</li>
                            </ul>
                        </div>
                    </div>
                </div>

                <div class="clinical-note">
                    <i class="fas fa-info-circle"></i>
                    <div>
                        <strong>Clinical Note:</strong> This AI analysis is based on historical prescribing patterns 
                        and is intended to support, not replace, clinical decision-making. All treatment decisions 
                        should consider the complete clinical picture and patient-specific factors.
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * 格式化风险因素名称
     */
    formatRiskFactorName(factor) {
        const nameMap = {
            'age_65_or_older': 'Age ≥65',
            'cardiovascular_disease': 'Cardiovascular Disease',
            'diabetes': 'Diabetes',
            'obesity': 'Obesity',
            'chronic_lung_disease': 'Chronic Lung Disease',
            'immunocompromised': 'Immunocompromised'
        };
        return nameMap[factor] || factor.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
}

// 创建全局实例
window.clinicalDS = new ClinicalDecisionSupport();

// 导出用于模块使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ClinicalDecisionSupport;
}

