import { useState } from 'react';
import { Activity, AlertCircle, Heart, ChevronRight, Info, Shield, Award, TrendingUp } from 'lucide-react';

function Main() {
    const [formData, setFormData] = useState({
        Pregnancies: 0,
        Glucose: 0,
        BloodPressure: 0,
        SkinThickness: 0,
        Insulin: 0,
        BMI: 0,
        DiabetesPedigree: 0,
        Age: 0
    });

    const [prediction, setPrediction] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: parseFloat(value) || 0
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setError(null);

        try {
            // Retrieve the token from localStorage
            const token = localStorage.getItem('access_token');
            if (!token) {
                throw new Error('Unauthorized: No access token found. Please log in.');
            }

            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Prediction failed');
            }

            setPrediction(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    const getRiskColor = (riskLevel) => {
        if (riskLevel === 'High Risk') return 'text-red-500';
        if (riskLevel === 'Medium Risk') return 'text-yellow-500';
        return 'text-green-500';
    };

    const getRiskBgColor = (riskLevel) => {
        if (riskLevel === 'High Risk') return 'bg-red-50 dark:bg-red-900/20 border-red-100 dark:border-red-800/50';
        if (riskLevel === 'Medium Risk') return 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-100 dark:border-yellow-800/50';
        return 'bg-green-50 dark:bg-green-900/20 border-green-100 dark:border-green-800/50';
    };

    const getRiskIcon = (riskLevel) => {
        if (riskLevel === 'High Risk') return <AlertCircle className="h-8 w-8 text-red-500" />;
        if (riskLevel === 'Medium Risk') return <Shield className="h-8 w-8 text-yellow-500" />;
        return <Award className="h-8 w-8 text-green-500" />;
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-blue-100 to-white dark:from-gray-900 dark:via-blue-950/30 dark:to-gray-800">
            <div className="max-w-7xl mx-auto px-4 py-12 sm:px-6 lg:px-8">
                {/* Header Section */}
                <div className="text-center mb-12">
                    <div className="inline-flex items-center px-4 py-2 mb-6 bg-blue-100/80 backdrop-blur-sm dark:bg-blue-900/40 text-blue-600 dark:text-blue-300 rounded-full shadow-sm">
                        <Heart className="h-5 w-5 mr-2 text-blue-600 dark:text-blue-400" />
                        <span className="text-sm font-medium">Diabetes Risk Assessment</span>
                    </div>
                    <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-4 tracking-tight">
                        Check Your <span className="text-blue-600 dark:text-blue-400">Diabetes Risk</span> Level
                    </h1>
                    <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
                        Enter your health metrics below for an AI-powered diabetes risk assessment
                    </p>
                </div>

                {/* Main Content */}
                <div className="grid lg:grid-cols-2 gap-8">
                    {/* Form Section */}
                    <div className="bg-white/80 backdrop-blur-sm dark:bg-gray-800/90 rounded-2xl shadow-xl p-8 border border-gray-200/80 dark:border-gray-700/80 transition-all hover:shadow-blue-100 dark:hover:shadow-blue-900/20">
                        <div className="flex items-center mb-6">
                            <Activity className="h-6 w-6 text-blue-600 dark:text-blue-400 mr-3" />
                            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                                Health Metrics
                            </h2>
                        </div>

                        {error && (
                            <div className="mb-6 bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-400 p-4 rounded-lg flex items-center border border-red-100 dark:border-red-800/50 shadow-sm">
                                <AlertCircle className="h-5 w-5 mr-2 flex-shrink-0" />
                                <span>{error}</span>
                            </div>
                        )}

                        <form onSubmit={handleSubmit} className="space-y-6">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                {Object.keys(formData).map((field) => (
                                    <div key={field} className="group">
                                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                                            {field.replace(/([A-Z])/g, ' $1').trim()}
                                        </label>
                                        <input
                                            type="number"
                                            name={field}
                                            value={formData[field]}
                                            onChange={handleInputChange}
                                            className="block w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-500 focus:border-transparent dark:bg-gray-700/80 dark:text-white shadow-sm transition-all hover:border-blue-300 dark:hover:border-blue-700"
                                            step="any"
                                            required
                                        />
                                    </div>
                                ))}
                            </div>

                            <button
                                type="submit"
                                disabled={isLoading}
                                className={`w-full py-4 px-6 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-all flex items-center justify-center shadow-md hover:shadow-lg ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                            >
                                {isLoading ? (
                                    <>
                                        <Activity className="animate-spin h-5 w-5 mr-2" />
                                        Analyzing...
                                    </>
                                ) : (
                                    <>
                                        Analyze Risk
                                        <ChevronRight className="ml-2 h-5 w-5" />
                                    </>
                                )}
                            </button>
                        </form>
                    </div>

                    {/* Results Section */}
                    <div className="bg-white/80 backdrop-blur-sm dark:bg-gray-800/90 rounded-2xl shadow-xl p-8 border border-gray-200/80 dark:border-gray-700/80 transition-all hover:shadow-blue-100 dark:hover:shadow-blue-900/20">
                        <div className="flex items-center mb-6">
                            <TrendingUp className="h-6 w-6 text-blue-600 dark:text-blue-400 mr-3" />
                            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                                Assessment Results
                            </h2>
                        </div>

                        {prediction ? (
                            <div className="space-y-8">
                                <div className={`${getRiskBgColor(prediction.risk)} p-6 rounded-xl border shadow-sm transition-all`}>
                                    <div className="flex items-center space-x-4">
                                        {getRiskIcon(prediction.risk)}
                                        <div>
                                            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-1">
                                                Risk Level
                                            </h3>
                                            <p className={`text-3xl font-bold ${getRiskColor(prediction.risk)}`}>
                                                {prediction.risk}
                                            </p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-blue-50 dark:bg-blue-900/20 p-6 rounded-xl border border-blue-100 dark:border-blue-800/50 shadow-sm">
                                    <div className="flex items-start">
                                        <Info className="h-6 w-6 text-blue-600 dark:text-blue-400 mt-0.5 mr-4 flex-shrink-0" />
                                        <div>
                                            <h4 className="text-lg font-medium text-blue-800 dark:text-blue-300 mb-2">
                                                Recommendations
                                            </h4>
                                            <p className="text-blue-700 dark:text-blue-400 whitespace-pre-line leading-relaxed">
                                                {prediction.recommendation}
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="h-full flex flex-col items-center justify-center py-16">
                                <div className="text-center text-gray-500 dark:text-gray-400 max-w-md">
                                    <div className="relative mx-auto w-24 h-24 mb-6">
                                        <Activity className="h-full w-full mx-auto text-blue-200 dark:text-blue-900/50" />
                                        <Heart className="h-12 w-12 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-blue-600 dark:text-blue-500 animate-pulse" />
                                    </div>
                                    <h3 className="text-xl font-medium text-gray-700 dark:text-gray-300 mb-3">No Results Yet</h3>
                                    <p className="text-gray-500 dark:text-gray-400">Enter your health metrics and click "Analyze Risk" to see your personalized assessment</p>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Main;