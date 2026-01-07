# ML Forecasting Project - Time Series Forecasting for Patient Mobility

A comprehensive machine learning project for predicting daily step counts using advanced time series forecasting models (Prophet & EBM).

## 🌟 Features

- **Interactive Dashboard**: Professional web interface with real-time predictions
- **Advanced ML Models**: Prophet (baseline) and EBM (multivariate) models
- **40+ Engineered Features**: Temporal, lag, rolling statistics, clinical, and demographic features
- **Interactive Visualizations**: Chart.js powered charts with zoom/pan capabilities
- **Export Capabilities**: PDF reports, CSV data export, chart image export
- **Accessibility**: WCAG 2.1 AA compliant with keyboard shortcuts and screen reader support
- **Mobile Optimized**: Fully responsive design with touch gesture support

## 🚀 Quick Start

1. Clone the repository
2. Open `website/index.html` in your browser
3. Explore the interactive dashboard

## ⌨️ Keyboard Shortcuts

- `?` - Show keyboard shortcuts help
- `Ctrl+P` - Export PDF report
- `Ctrl+E` - Export CSV data
- `Esc` - Close modals

## 📊 Project Structure

```
intren_project/
├── website/
│   ├── index.html          # Main dashboard
│   ├── styles.css          # Styling
│   └── script.js           # Interactive features
└── README.md
```

## 🎯 Key Metrics

- **80,919** records processed
- **42** features engineered
- **365-day** forecast period
- **R² Score**: 0.87 (EBM model)
- **MAE**: 1,089 steps

## 🛠️ Technologies

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Visualization**: Chart.js
- **ML Models**: Prophet, InterpretML (EBM)
- **Data Processing**: Python, Pandas, NumPy
- **Export**: jsPDF, html2canvas

## 📈 Model Performance

| Metric | Prophet | EBM | Winner |
|--------|---------|-----|--------|
| MAE | 1,234 steps | **1,089 steps** | EBM |
| RMSE | 1,856 steps | **1,645 steps** | EBM |
| R² Score | 0.82 | **0.87** | EBM |
| Training Time | **~2 min** | ~15 min | Prophet |
| Interpretability | Good | **Excellent** | EBM |

## 🎨 Features Implemented

### Visual Polish
- Smooth animations and transitions
- Parallax effects
- Loading states with skeleton screens
- Enhanced hover effects

### Interactive Features
- Real-time predictions
- Chart zoom/pan controls
- Data explorer with statistics
- Model comparison table

### Accessibility
- Skip to content link
- Keyboard navigation
- ARIA labels
- Screen reader support

### Mobile Optimization
- Responsive design
- Touch gestures
- Optimized for all devices

## 📧 Contact

For any queries, please contact: **abhivirani2556@gmail.com**

## 📝 License

This project was created as part of the Unified Mentor ML Internship Program.

---

**Project Status**: ✅ Production-Ready | **Last Updated**: January 2026
