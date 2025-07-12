"""
CSS styles for the Streamlit application
"""

MAIN_STYLES = """
<style>
/* Main app styling */
.main-header {
    font-size: 3rem;
    font-weight: 700;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #1f77b4, #17becf);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.sub-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #333;
    margin: 1.5rem 0 1rem 0;
    border-bottom: 2px solid #1f77b4;
    padding-bottom: 0.5rem;
}

/* Metric cards */
.metric-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}

.metric-container:hover {
    transform: translateY(-5px);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
}

.metric-label {
    font-size: 1rem;
    opacity: 0.9;
    margin: 0;
}

/* Status boxes */
.success-box {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
}

.warning-box {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(250, 112, 154, 0.3);
}

.info-box {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: #333;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(168, 237, 234, 0.3);
}

.error-box {
    background: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
    padding: 1.5rem;
    border-radius: 15px;
    color: white;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
}

/* Image containers */
.image-container {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin: 1rem 0;
    transition: transform 0.3s ease;
}

.image-container:hover {
    transform: scale(1.02);
}

.image-caption {
    background: rgba(0,0,0,0.8);
    color: white;
    padding: 0.5rem 1rem;
    text-align: center;
    font-weight: 500;
}

/* Sidebar styling */
.sidebar .stButton > button {
    width: 100%;
    border-radius: 25px;
    border: none;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 600;
    padding: 0.75rem 1rem;
    transition: all 0.3s ease;
}

.sidebar .stButton > button:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}

.sidebar .stSelectbox > div > div {
    border-radius: 15px;
    border: 2px solid #e0e0e0;
}

.sidebar .stSlider > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2);
}

/* Progress bars */
.progress-container {
    background: #f0f0f0;
    border-radius: 25px;
    overflow: hidden;
    height: 30px;
    margin: 1rem 0;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 25px;
    transition: width 0.5s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 600;
}

/* Cards */
.card {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border: 1px solid #e0e0e0;
}

.card-header {
    font-size: 1.2rem;
    font-weight: 600;
    color: #333;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #f0f0f0;
}

/* Tables */
.dataframe {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.dataframe th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 600;
    padding: 1rem;
}

.dataframe td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #f0f0f0;
}

/* Footer */
.footer {
    text-align: center;
    padding: 3rem 2rem;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 15px;
    margin-top: 3rem;
    border: 1px solid #e0e0e0;
}

.footer h3 {
    color: #333;
    margin-bottom: 1rem;
}

.footer a {
    color: #1f77b4;
    text-decoration: none;
    font-weight: 600;
    margin: 0 1rem;
    transition: color 0.3s ease;
}

.footer a:hover {
    color: #17becf;
    text-decoration: underline;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideIn {
    from { transform: translateX(-100%); }
    to { transform: translateX(0); }
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.fade-in {
    animation: fadeIn 0.5s ease-out;
}

.slide-in {
    animation: slideIn 0.5s ease-out;
}

/* Loading spinner */
.loading-spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #1f77b4;
    border-radius: 50%;
    width: 50px;
    height: 50px;
    animation: spin 1s linear infinite;
    margin: 2rem auto;
}

/* Responsive design */
@media (max-width: 768px) {
    .main-header {
        font-size: 2rem;
    }
    
    .metric-container {
        padding: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
    }
    
    .card {
        padding: 1rem;
    }
}

/* Dark theme support */
@media (prefers-color-scheme: dark) {
    .card {
        background: #2d3748;
        border-color: #4a5568;
    }
    
    .card-header {
        color: #e2e8f0;
        border-color: #4a5568;
    }
    
    .info-box {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        color: #e2e8f0;
    }
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

/* Streamlit specific overrides */
.stApp {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
}

.css-1d391kg {
    padding-top: 1rem;
}

.css-18e3th9 {
    padding-top: 0;
}

/* Hide Streamlit menu and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom file uploader */
.stFileUploader > div {
    border-radius: 15px;
    border: 2px dashed #1f77b4;
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    padding: 2rem;
    text-align: center;
}

.stFileUploader > div:hover {
    border-color: #17becf;
    background: linear-gradient(135deg, #e2e8f0 0%, #f8fafc 100%);
}

/* Custom selectbox */
.stSelectbox > div > div {
    border-radius: 15px;
    border: 2px solid #e2e8f0;
}

.stSelectbox > div > div:focus-within {
    border-color: #1f77b4;
    box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.2);
}

/* Custom slider */
.stSlider > div > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 15px;
}

.stSlider > div > div > div > div {
    background: white;
    border: 3px solid #1f77b4;
    box-shadow: 0 2px 10px rgba(31, 119, 180, 0.3);
}
</style>
"""

DARK_THEME_STYLES = """
<style>
/* Dark theme styles */
.dark-theme {
    background: #1a202c;
    color: #e2e8f0;
}

.dark-theme .main-header {
    color: #63b3ed;
    background: linear-gradient(90deg, #63b3ed, #4fd1c7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.dark-theme .sub-header {
    color: #e2e8f0;
    border-color: #63b3ed;
}

.dark-theme .card {
    background: #2d3748;
    border-color: #4a5568;
    color: #e2e8f0;
}

.dark-theme .card-header {
    color: #e2e8f0;
    border-color: #4a5568;
}

.dark-theme .info-box {
    background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
    color: #e2e8f0;
}

.dark-theme .footer {
    background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
    color: #e2e8f0;
}

.dark-theme .footer a {
    color: #63b3ed;
}

.dark-theme .footer a:hover {
    color: #4fd1c7;
}
</style>
"""
