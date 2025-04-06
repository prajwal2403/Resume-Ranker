import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertCircle, ChevronRight, User, Lock, Activity, Heart, Award, BarChart2, Shield, Info, Clipboard, X, ChevronDown, LogIn, UserPlus } from 'lucide-react';

function HeroSection() {
    const navigate = useNavigate();
    const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
    const [showModal, setShowModal] = useState(false);
    const [isLoading, setIsLoading] = useState(false);

    // Sample patient data sets
    const samplePatients = {
        lowRisk: {
            age: 32,
            bloodPressure: "110/75",
            cholesterol: 180,
            bloodSugar: 95,
            bmi: 22.3,
            smoking: false,
            familyHistory: false
        },
        moderateRisk: {
            age: 45,
            bloodPressure: "130/85",
            cholesterol: 210,
            bloodSugar: 120,
            bmi: 27.5,
            smoking: false,
            familyHistory: true
        },
        highRisk: {
            age: 58,
            bloodPressure: "150/95",
            cholesterol: 260,
            bloodSugar: 145,
            bmi: 32.1,
            smoking: true,
            familyHistory: true
        }
    };

    const [patientData, setPatientData] = useState(samplePatients.moderateRisk);
    const [analysisResults, setAnalysisResults] = useState(null);

    // Perform analysis with loading state
    const performAnalysis = () => {
        setIsLoading(true);

        // Simulate API call delay
        setTimeout(() => {
            const riskScore = calculateRiskScore(patientData);

            setAnalysisResults({
                riskScore,
                riskLevel: getRiskLevel(riskScore),
                recommendations: generateRecommendations(patientData, riskScore),
                timestamp: new Date().toLocaleString()
            });

            setShowModal(true);
            setIsLoading(false);
        }, 1500);
    };

    // Load sample data
    const loadSampleData = (riskLevel) => {
        setPatientData(samplePatients[riskLevel]);
    };

    // Get risk level classification
    const getRiskLevel = (score) => {
        if (score > 70) return "High";
        if (score > 40) return "Moderate";
        return "Low";
    };

    // Navigation handlers
    const handleLoginClick = (e) => {
        e.stopPropagation();
        setIsUserMenuOpen(false);
        navigate('/login');
    };

    const handleSignupClick = (e) => {
        e.stopPropagation();
        setIsUserMenuOpen(false);
        navigate('/signup');
    };

    // Risk calculation
    const calculateRiskScore = (data) => {
        let score = 0;

        // Age factor
        if (data.age > 60) score += 25;
        else if (data.age > 50) score += 20;
        else if (data.age > 40) score += 15;
        else if (data.age > 30) score += 5;

        // Blood pressure
        const [systolic, diastolic] = data.bloodPressure.split('/').map(Number);
        if (systolic > 160 || diastolic > 100) score += 25;
        else if (systolic > 140 || diastolic > 90) score += 15;
        else if (systolic > 120 || diastolic > 80) score += 5;

        // Cholesterol
        if (data.cholesterol > 240) score += 20;
        else if (data.cholesterol > 200) score += 10;

        // Blood sugar
        if (data.bloodSugar > 126) score += 20;
        else if (data.bloodSugar > 100) score += 10;

        // BMI
        if (data.bmi > 30) score += 15;
        else if (data.bmi > 25) score += 10;

        // Other factors
        if (data.smoking) score += 15;
        if (data.familyHistory) score += 10;

        return Math.min(score, 100);
    };

    // Generate recommendations
    const generateRecommendations = (data, score) => {
        const recommendations = [];
        const systolic = parseInt(data.bloodPressure.split('/')[0]);

        // General recommendations based on risk level
        if (score > 70) {
            recommendations.push("Schedule a doctor's appointment immediately");
            recommendations.push("Consider comprehensive blood work and ECG");
        } else if (score > 40) {
            recommendations.push("Schedule a follow-up within 1 month");
        }

        // Blood pressure specific
        if (systolic > 140) {
            recommendations.push("Daily blood pressure monitoring required");
            recommendations.push("Reduce sodium intake to <1500mg/day");
        } else if (systolic > 120) {
            recommendations.push("Weekly blood pressure checks recommended");
        }

        // Cholesterol specific
        if (data.cholesterol > 240) {
            recommendations.push("Consider statin therapy after doctor consultation");
            recommendations.push("Increase soluble fiber intake (oats, beans, apples)");
        } else if (data.cholesterol > 200) {
            recommendations.push("Replace saturated fats with unsaturated fats");
        }

        // BMI specific
        if (data.bmi > 30) {
            recommendations.push("Aim for 5-10% weight loss over 6 months");
            recommendations.push("150 minutes of moderate exercise weekly");
        } else if (data.bmi > 25) {
            recommendations.push("Maintain weight through balanced diet");
        }

        // Lifestyle factors
        if (data.smoking) {
            recommendations.push("Quit smoking - consider nicotine replacement therapy");
        }
        if (data.familyHistory) {
            recommendations.push("Annual comprehensive health screening recommended");
        }

        // Add positive reinforcement
        if (score < 40) {
            recommendations.push("Maintain current healthy lifestyle habits");
        }

        return recommendations;
    };

    // Get risk color
    const getRiskColor = (score) => {
        if (score > 70) return "bg-red-500 text-white";
        if (score > 40) return "bg-amber-500 text-white";
        return "bg-green-500 text-white";
    };

    // Close dropdown when clicking outside
    const handleClickOutside = (e) => {
        const dropdownContainer = document.querySelector('.dropdown-container');
        if (isUserMenuOpen && dropdownContainer && !dropdownContainer.contains(e.target)) {
            setIsUserMenuOpen(false);
        }
    };

    // Toggle user menu
    const toggleUserMenu = () => {
        setIsUserMenuOpen(!isUserMenuOpen);
    };

    return (
        <div className="bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-800 min-h-screen" onClick={handleClickOutside}>
            {/* Navigation bar */}
            <nav className="bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm border-b border-gray-200 dark:border-gray-800 sticky top-0 z-10">
                <div className="max-w-screen-xl flex flex-wrap items-center justify-between mx-auto p-4">
                    <a href="#" className="flex items-center">
                        <Heart className="h-8 w-8 mr-2 text-blue-600 dark:text-blue-400" />
                        <span className="self-center text-2xl font-semibold whitespace-nowrap dark:text-white">HealthML</span>
                    </a>

                    <div className="flex md:order-2 relative dropdown-container">
                        <button
                            onClick={toggleUserMenu}
                            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors flex items-center"
                        >
                            <User className="h-5 w-5 mr-2" />
                            <span>Account</span>
                            <ChevronDown className={`ml-1 h-4 w-4 transition-transform ${isUserMenuOpen ? 'rotate-180' : ''}`} />
                        </button>

                        {/* User dropdown menu */}
                        {isUserMenuOpen && (
                            <div className="absolute right-0 top-full mt-2 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 z-20">
                                <ul className="py-2">
                                    <li>
                                        <button
                                            onClick={handleLoginClick}
                                            className="flex items-center w-full px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700"
                                        >
                                            <LogIn className="h-4 w-4 mr-2" />
                                            <span>Login</span>
                                        </button>
                                    </li>
                                    <li>
                                        <button
                                            onClick={handleSignupClick}
                                            className="flex items-center w-full px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700"
                                        >
                                            <UserPlus className="h-4 w-4 mr-2" />
                                            <span>Sign Up</span>
                                        </button>
                                    </li>
                                </ul>
                            </div>
                        )}
                    </div>
                </div>
            </nav>

            {/* Hero Section */}
            <section className="relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-50 via-white to-blue-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 z-0"></div>
                <div className="absolute top-20 right-20 w-64 h-64 bg-blue-400/10 dark:bg-blue-600/10 rounded-full blur-3xl"></div>
                <div className="absolute bottom-10 left-10 w-96 h-96 bg-purple-400/10 dark:bg-purple-600/10 rounded-full blur-3xl"></div>

                <div className="max-w-screen-xl mx-auto px-4 py-16 md:py-24 relative z-10">
                    <div className="flex flex-col lg:flex-row items-center">
                        <div className="lg:w-1/2 lg:pr-12 mb-10 lg:mb-0">
                            <div className="inline-flex items-center px-4 py-2 mb-6 bg-blue-100 dark:bg-blue-900/40 text-blue-600 dark:text-blue-300 rounded-full">
                                <span className="flex items-center justify-center w-6 h-6 mr-2 bg-blue-600 dark:bg-blue-500 text-white text-xs font-bold rounded-full">!</span>
                                <p className="text-sm font-medium">HealthML Pro is now available</p>
                                <ChevronRight size={16} className="ml-2" />
                            </div>

                            <h1 className="text-4xl md:text-5xl font-extrabold mb-6 text-gray-900 dark:text-white leading-tight">
                                <span className="block">AI-Powered Health</span>
                                <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Risk Assessment</span>
                            </h1>

                            <p className="text-lg text-gray-600 dark:text-gray-300 mb-8 max-w-xl">
                                Our advanced machine learning algorithms analyze your health data to provide personalized risk assessments and evidence-based recommendations.
                            </p>

                            <div className="flex flex-col sm:flex-row gap-4 mb-6">
                                <button
                                    onClick={performAnalysis}
                                    disabled={isLoading}
                                    className="px-6 py-3 text-white bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition-colors flex items-center justify-center shadow-lg hover:shadow-xl transform hover:-translate-y-1 disabled:opacity-75 disabled:cursor-not-allowed"
                                >
                                    {isLoading ? (
                                        <>
                                            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                            </svg>
                                            Analyzing...
                                        </>
                                    ) : (
                                        <>
                                            Analyze My Health
                                            <ChevronRight className="ml-2" />
                                        </>
                                    )}
                                </button>

                                <button
                                    onClick={() => setShowModal(true)}
                                    className="px-6 py-3 text-gray-700 dark:text-white bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg font-medium transition-colors flex items-center justify-center"
                                >
                                    <Info className="mr-2" size={18} />
                                    View Sample Report
                                </button>
                            </div>

                            <div className="flex flex-wrap gap-2">
                                <button
                                    onClick={() => loadSampleData('lowRisk')}
                                    className="px-4 py-2 text-xs font-medium text-green-700 bg-green-100 hover:bg-green-200 rounded-lg transition-colors"
                                >
                                    Load Low Risk Sample
                                </button>
                                <button
                                    onClick={() => loadSampleData('moderateRisk')}
                                    className="px-4 py-2 text-xs font-medium text-amber-700 bg-amber-100 hover:bg-amber-200 rounded-lg transition-colors"
                                >
                                    Load Moderate Risk Sample
                                </button>
                                <button
                                    onClick={() => loadSampleData('highRisk')}
                                    className="px-4 py-2 text-xs font-medium text-red-700 bg-red-100 hover:bg-red-200 rounded-lg transition-colors"
                                >
                                    Load High Risk Sample
                                </button>
                            </div>
                        </div>

                        <div className="lg:w-1/2">
                            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6 border border-gray-200 dark:border-gray-700">
                                <div className="flex items-center justify-between mb-6">
                                    <h2 className="text-xl font-bold text-gray-900 dark:text-white">Health Metrics Dashboard</h2>
                                    <div className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-300 text-xs font-medium px-3 py-1 rounded-full">Demo</div>
                                </div>

                                <div className="grid grid-cols-2 gap-4 mb-6">
                                    <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                                        <div className="flex items-center mb-2">
                                            <Activity className="h-5 w-5 text-blue-600 dark:text-blue-400 mr-2" />
                                            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-300">Blood Pressure</h3>
                                        </div>
                                        <p className="text-2xl font-bold text-gray-900 dark:text-white">{patientData.bloodPressure}</p>
                                    </div>

                                    <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                                        <div className="flex items-center mb-2">
                                            <Heart className="h-5 w-5 text-blue-600 dark:text-blue-400 mr-2" />
                                            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-300">Cholesterol</h3>
                                        </div>
                                        <p className="text-2xl font-bold text-gray-900 dark:text-white">{patientData.cholesterol} mg/dL</p>
                                    </div>

                                    <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                                        <div className="flex items-center mb-2">
                                            <BarChart2 className="h-5 w-5 text-blue-600 dark:text-blue-400 mr-2" />
                                            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-300">BMI</h3>
                                        </div>
                                        <p className="text-2xl font-bold text-gray-900 dark:text-white">{patientData.bmi}</p>
                                    </div>

                                    <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                                        <div className="flex items-center mb-2">
                                            <User className="h-5 w-5 text-blue-600 dark:text-blue-400 mr-2" />
                                            <h3 className="text-sm font-medium text-gray-600 dark:text-gray-300">Age</h3>
                                        </div>
                                        <p className="text-2xl font-bold text-gray-900 dark:text-white">{patientData.age}</p>
                                    </div>
                                </div>

                                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-100 dark:border-blue-800/50 flex items-start">
                                    <Shield className="h-6 w-6 text-blue-600 dark:text-blue-400 mr-3 mt-0.5 flex-shrink-0" />
                                    <div>
                                        <h3 className="text-sm font-medium text-blue-800 dark:text-blue-300 mb-1">Your data is secure</h3>
                                        <p className="text-xs text-blue-700 dark:text-blue-400">Health data is encrypted and never shared with third parties.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section className="py-16 bg-white dark:bg-gray-800">
                <div className="max-w-screen-xl mx-auto px-4">
                    <div className="text-center mb-12">
                        <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">Advanced Features</h2>
                        <p className="text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">Discover how our AI-powered platform can help you make informed decisions about your health.</p>
                    </div>

                    <div className="grid md:grid-cols-3 gap-8">
                        {[
                            {
                                icon: <Activity className="h-6 w-6 text-blue-600 dark:text-blue-400" />,
                                title: "Personal Risk Assessment",
                                description: "Get personalized health risk analysis based on your specific health metrics and genetic factors.",
                                color: "blue"
                            },
                            {
                                icon: <Clipboard className="h-6 w-6 text-purple-600 dark:text-purple-400" />,
                                title: "Actionable Recommendations",
                                description: "Receive evidence-based suggestions to improve your health outcomes and reduce risk factors.",
                                color: "purple"
                            },
                            {
                                icon: <Award className="h-6 w-6 text-green-600 dark:text-green-400" />,
                                title: "Progress Tracking",
                                description: "Monitor your health improvements over time with detailed metrics and visualizations.",
                                color: "green"
                            }
                        ].map((feature, index) => (
                            <div key={index} className="bg-gray-50 dark:bg-gray-700 rounded-xl p-6 border border-gray-200 dark:border-gray-600 transition-transform hover:scale-105">
                                <div className={`w-12 h-12 bg-${feature.color}-100 dark:bg-${feature.color}-900/50 rounded-full flex items-center justify-center mb-4`}>
                                    {feature.icon}
                                </div>
                                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">{feature.title}</h3>
                                <p className="text-gray-600 dark:text-gray-300">{feature.description}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Results Modal */}
            {showModal && (
                <div className="fixed inset-0 flex items-center justify-center z-50 bg-black/50 backdrop-blur-sm">
                    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl max-w-md w-full max-h-[90vh] overflow-y-auto mx-4">
                        <div className="p-6">
                            <div className="flex justify-between items-center mb-4">
                                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Health Analysis Report</h2>
                                <button
                                    onClick={() => setShowModal(false)}
                                    className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                                >
                                    <X size={24} />
                                </button>
                            </div>

                            {analysisResults ? (
                                <>
                                    <div className="mb-6">
                                        <div className="flex justify-between items-center mb-2">
                                            <h3 className="text-lg font-medium text-gray-700 dark:text-gray-300">Risk Score</h3>
                                            <div className={`px-3 py-1 rounded-full text-sm font-bold ${getRiskColor(analysisResults.riskScore)}`}>
                                                {analysisResults.riskLevel} Risk
                                            </div>
                                        </div>

                                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5 mb-3">
                                            <div
                                                className={`h-2.5 rounded-full ${analysisResults.riskScore > 70 ? "bg-red-500" :
                                                    analysisResults.riskScore > 40 ? "bg-amber-500" : "bg-green-500"
                                                    }`}
                                                style={{ width: `${analysisResults.riskScore}%` }}
                                            ></div>
                                        </div>

                                        <p className="text-sm text-gray-500 dark:text-gray-400 text-right">
                                            Score: {analysisResults.riskScore}/100
                                        </p>
                                    </div>

                                    <div className="mb-6">
                                        <h3 className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-3">Key Recommendations</h3>
                                        <ul className="space-y-3">
                                            {analysisResults.recommendations.map((rec, index) => (
                                                <li key={index} className="flex items-start">
                                                    <div className={`flex-shrink-0 mt-1 mr-3 ${analysisResults.riskScore > 70 ? "text-red-500" : analysisResults.riskScore > 40 ? "text-amber-500" : "text-green-500"}`}>
                                                        {analysisResults.riskScore > 70 ? (
                                                            <AlertCircle className="h-5 w-5" />
                                                        ) : (
                                                            <CheckCircle className="h-5 w-5" />
                                                        )}
                                                    </div>
                                                    <span className="text-gray-600 dark:text-gray-300">{rec}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>

                                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-4">
                                        <p>Report generated: {analysisResults.timestamp}</p>
                                        <p className="mt-1">This assessment is for informational purposes only and not a substitute for professional medical advice.</p>
                                    </div>
                                </>
                            ) : (
                                <div className="text-center py-8">
                                    <p className="text-gray-600 dark:text-gray-300">No analysis data available. Please run an analysis first.</p>
                                </div>
                            )}

                            <button
                                onClick={() => setShowModal(false)}
                                className="w-full mt-6 py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
                            >
                                Close Report
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

// CheckCircle component
const CheckCircle = ({ className, size }) => {
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width={size || 24}
            height={size || 24}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={className}
        >
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
            <polyline points="22 4 12 14.01 9 11.01"></polyline>
        </svg>
    );
};

export default HeroSection;