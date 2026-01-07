# 🏥 ML Forecasting Project - Time Series Forecasting for Patient Mobility

> **Advanced machine learning dashboard for predicting daily step counts using Prophet & EBM models**

[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](https://github.com/AbhiGuru25/ML-forecasting-project)
[![ML Models](https://img.shields.io/badge/ML-Prophet%20%7C%20EBM-blue)](https://github.com/AbhiGuru25/ML-forecasting-project)
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/AbhiGuru25/ML-forecasting-project)

---

## � Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Model Performance](#model-performance)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)

---

## 🎯 Overview

This project implements an **interactive web-based dashboard** for predicting patient mobility using advanced time series forecasting techniques. Built as part of the Unified Mentor ML Internship Program, it combines cutting-edge machine learning models with a professional, accessible user interface.

### Key Highlights

- 📊 **80,919** records processed from patient mobility data
- 🔧 **42** engineered features (temporal, lag, rolling stats, clinical, demographic)
- 📅 **365-day** forecast horizon
- 🎯 **87% R² Score** with EBM model
- 📉 **1,089 steps MAE** - highly accurate predictions
- 🌐 **100% client-side** - no server required

---

## ✨ Features

### 🎨 Interactive Dashboard

- **Real-time Predictions**: Generate forecasts instantly with customizable parameters
- **Interactive Visualizations**: Zoom, pan, and explore charts powered by Chart.js
- **Data Explorer**: View detailed statistics and insights from the dataset
- **Model Comparison**: Side-by-side comparison of Prophet vs EBM performance

### 📤 Export Capabilities

- **PDF Reports**: Generate comprehensive reports with jsPDF
- **CSV Data Export**: Download forecast data for further analysis
- **Chart Images**: Export visualizations as PNG images with html2canvas

### ♿ Accessibility Features

- **WCAG 2.1 AA Compliant**: Fully accessible to users with disabilities
- **Keyboard Navigation**: Complete keyboard support with shortcuts
- **Screen Reader Support**: ARIA labels and semantic HTML
- **Skip Links**: Quick navigation for keyboard users

### 📱 Mobile Optimization

- **Fully Responsive**: Adapts seamlessly to all screen sizes
- **Touch Gestures**: Swipe and pinch-to-zoom support
- **Optimized Performance**: Fast loading on mobile devices
- **Touch-Friendly UI**: Large buttons and intuitive interactions

### 🎭 Visual Polish

- **Smooth Animations**: Fade-in effects and transitions
- **Parallax Scrolling**: Engaging hero section
- **Loading States**: Skeleton screens and spinners
- **Dark Theme**: Professional dark mode interface
- **Hover Effects**: Interactive card lifts and shadows

---

## 🚀 Quick Start

### Option 1: Direct Use (No Installation)

1. **Clone the repository**
   ```bash
   git clone https://github.com/AbhiGuru25/ML-forecasting-project.git
   cd ML-forecasting-project
   ```

2. **Open the dashboard**
   ```bash
   # Simply open website/index.html in your browser
   # Or use a local server (recommended):
   python -m http.server 8000
   # Then visit: http://localhost:8000/website/
   ```

3. **Explore the features**
   - Try the prediction form
   - Export data as PDF/CSV
   - Press `?` to see keyboard shortcuts

### Option 2: GitHub Pages (Live Demo)

####Visit the live demo at: https://ml-forecasting-project.onrender.com
---

## 📁 Project Structure

```
ML-forecasting-project/
├── website/                    # Interactive Dashboard
│   ├── index.html             # Main dashboard page
│   ├── styles.css             # Comprehensive styling
│   └── script.js              # Interactive features & ML integration
│
├── src/                       # Python Source Code
│   ├── preprocessing.py       # Data preprocessing pipeline
│   ├── feature_engineering.py # Feature creation
│   ├── data_loader.py         # Data loading utilities
│   ├── cloud_utils.py         # Cloud integration (AWS S3)
│   ├── explainability.py      # Model interpretability
│   ├── forecast_output.py     # Forecast generation
│   └── models/                # ML Models
│       ├── baseline_model.py  # Prophet implementation
│       └── multivariate_model.py  # EBM implementation
│
├── docs/                      # Documentation
│   ├── PROJECT_OVERVIEW.md    # Detailed project overview
│   ├── QUICKSTART.md          # Quick start guide
│   ├── SUBMISSION_GUIDE.md    # Submission instructions
│   └── presentations/         # Presentation materials
│
├── tests/                     # Unit tests
├── README.md                  # This file
├── requirements.txt           # Python dependencies
└── .gitignore                # Git ignore rules
```

---

## 🛠️ Technologies

### Frontend Stack

| Technology | Purpose | Version |
|-----------|---------|---------|
| **HTML5** | Structure & Semantics | Latest |
| **CSS3** | Styling & Animations | Latest |
| **JavaScript (ES6+)** | Interactivity & Logic | ES2021 |
| **Chart.js** | Data Visualization | 3.9.1 |
| **jsPDF** | PDF Generation | 2.5.1 |
| **html2canvas** | Chart Export | 1.4.1 |
| **Flatpickr** | Date Picker | 4.6.13 |

### Backend/ML Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Core programming language |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computing |
| **Prophet** | Time series forecasting (baseline) |
| **InterpretML (EBM)** | Explainable boosting machine |
| **Scikit-learn** | ML utilities & metrics |
| **Matplotlib/Seaborn** | Visualization |
| **boto3** | AWS S3 integration |

---

## 📈 Model Performance

### Comprehensive Comparison

| Metric | Prophet (Baseline) | EBM (Advanced) | Improvement | Winner |
|--------|-------------------|----------------|-------------|--------|
| **MAE** | 1,234 steps | **1,089 steps** | ↓ 11.7% | 🏆 EBM |
| **RMSE** | 1,856 steps | **1,645 steps** | ↓ 11.4% | 🏆 EBM |
| **R² Score** | 0.82 | **0.87** | ↑ 6.1% | 🏆 EBM |
| **MAPE** | 15.2% | **13.1%** | ↓ 13.8% | 🏆 EBM |
| **Training Time** | **~2 min** | ~15 min | - | 🏆 Prophet |
| **Interpretability** | Good | **Excellent** | - | 🏆 EBM |
| **Multivariate** | ❌ No | ✅ **Yes (42 features)** | - | 🏆 EBM |

### Model Insights

**Prophet (Baseline)**
- ✅ Fast training (~2 minutes)
- ✅ Good for univariate time series
- ✅ Automatic seasonality detection
- ❌ Cannot leverage clinical features
- ❌ Lower accuracy

**EBM (Advanced)**
- ✅ Excellent accuracy (R² = 0.87)
- ✅ Leverages 42 engineered features
- ✅ Highly interpretable (feature importance)
- ✅ Captures complex patterns
- ❌ Longer training time (~15 minutes)

**Conclusion:** EBM is the recommended model for production deployment due to superior accuracy and interpretability, despite longer training times.

---

## ⌨️ Keyboard Shortcuts

Enhance your productivity with these keyboard shortcuts:

| Shortcut | Action |
|----------|--------|
| `?` | Show keyboard shortcuts help modal |
| `Ctrl + P` (or `Cmd + P`) | Export comprehensive PDF report |
| `Ctrl + E` (or `Cmd + E`) | Export forecast data as CSV |
| `Ctrl + K` (or `Cmd + K`) | Focus search bar (future feature) |
| `Esc` | Close modals and overlays |
| `Tab` | Navigate between interactive elements |

---

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Git

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AbhiGuru25/ML-forecasting-project.git
   cd ML-forecasting-project
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   # Option 1: Direct file open
   # Open website/index.html in your browser
   
   # Option 2: Local server (recommended)
   python -m http.server 8000
   # Visit: http://localhost:8000/website/
   ```

---

## 📖 Usage

### Making Predictions

1. Navigate to the **Predictions** section
2. Enter patient parameters:
   - Age
   - Gender
   - Active therapies
   - Side effect intensity
3. Click **Generate Forecast**
4. View results with confidence intervals

### Exporting Data

**PDF Report:**
- Click "Export as PDF" or press `Ctrl+P`
- Includes overview, metrics, and visualizations

**CSV Data:**
- Click "Export as CSV" or press `Ctrl+E`
- Downloads forecast data for analysis

**Chart Images:**
- Click "Export Charts"
- Saves all visualizations as PNG files

### Exploring Data

1. Visit the **Data Explorer** section
2. View key statistics and distributions
3. Filter and search through the dataset
4. Analyze feature importance

---

## 🎓 Learning Outcomes

This project demonstrates:

1. **Advanced ML Techniques**: Time series forecasting with Prophet & EBM
2. **Feature Engineering**: Creating 42 meaningful features from raw data
3. **Web Development**: Building interactive dashboards with vanilla JS
4. **Accessibility**: WCAG 2.1 AA compliance and keyboard navigation
5. **Data Visualization**: Interactive charts with Chart.js
6. **Code Organization**: Modular, maintainable code structure
7. **Documentation**: Comprehensive README and inline comments

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Abhi Virani**  
📧 Email: [abhivirani2556@gmail.com](mailto:abhivirani2556@gmail.com)  
🔗 GitHub: [@AbhiGuru25](https://github.com/AbhiGuru25)  
💼 LinkedIn: [Connect with me](https://www.linkedin.com/in/abhi-virani-a37138294/)

---

## 📝 License

This project was created as part of the Internship Program

---

## 🙏 Acknowledgments

- **Unified Mentor** for the internship opportunity
- **Prophet** by Facebook Research
- **InterpretML** by Microsoft Research
- **Chart.js** community for excellent documentation

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

**Status:** ✅ Production-Ready | **Last Updated:** January 2026

Made with ❤️ by [Abhi Virani](https://github.com/AbhiGuru25)

</div>

