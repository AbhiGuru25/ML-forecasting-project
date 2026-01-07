"""
Explainability Module
Provides interpretability analysis for forecasting models.
"""

import pandas as pd
import numpy as np
from interpret import show
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExplainer:
    """Provides explainability for forecasting models."""
    
    def __init__(self, model, model_type: str = 'ebm'):
        """
        Initialize explainer.
        
        Args:
            model: Trained model (EBM or other)
            model_type: Type of model ('ebm', 'xgboost', etc.)
        """
        self.model = model
        self.model_type = model_type
        
    def get_global_explanation(self) -> Dict:
        """
        Get global feature importance.
        
        Returns:
            Dictionary with feature names and importance scores
        """
        logger.info("Generating global explanation...")
        
        if self.model_type == 'ebm':
            # Use EBM's built-in global explanation
            ebm_global = self.model.explain_global()
            
            explanation = {
                'names': ebm_global.data()['names'],
                'scores': ebm_global.data()['scores']
            }
            
            logger.info(f"✓ Global explanation generated for {len(explanation['names'])} features")
            
            return explanation
        else:
            logger.warning(f"Global explanation not implemented for {self.model_type}")
            return {}
    
    def plot_feature_importance(self, top_n: int = 15, save_path: str = None) -> None:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            save_path: Optional path to save plot
        """
        logger.info(f"Plotting top {top_n} feature importances...")
        
        explanation = self.get_global_explanation()
        
        if not explanation:
            logger.warning("No explanation available")
            return
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': explanation['names'],
            'Importance': explanation['scores']
        })
        
        # Sort and get top N
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y='Feature', x='Importance', palette='viridis')
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"  Plot saved to {save_path}")
        
        plt.close()
        
        logger.info("✓ Feature importance plot created")
    
    def analyze_categorical_impact(self, feature_name: str) -> Dict:
        """
        Analyze impact of a categorical variable.
        
        Args:
            feature_name: Name of the categorical feature
        
        Returns:
            Dictionary with impact analysis
        """
        logger.info(f"Analyzing impact of categorical variable: {feature_name}")
        
        if self.model_type == 'ebm':
            ebm_global = self.model.explain_global()
            
            # Find the feature
            names = ebm_global.data()['names']
            if feature_name in names:
                idx = names.index(feature_name)
                score = ebm_global.data()['scores'][idx]
                
                impact = {
                    'feature': feature_name,
                    'importance_score': score,
                    'interpretation': self._interpret_score(score)
                }
                
                logger.info(f"  Importance score: {score:.4f}")
                logger.info(f"  Interpretation: {impact['interpretation']}")
                
                return impact
            else:
                logger.warning(f"Feature {feature_name} not found in model")
                return {}
        
        return {}
    
    def _interpret_score(self, score: float) -> str:
        """
        Interpret importance score.
        
        Args:
            score: Importance score
        
        Returns:
            Human-readable interpretation
        """
        if score > 0.1:
            return "High impact - This feature significantly influences step count predictions"
        elif score > 0.05:
            return "Moderate impact - This feature has noticeable influence on predictions"
        elif score > 0.01:
            return "Low impact - This feature has minor influence on predictions"
        else:
            return "Minimal impact - This feature has very little influence on predictions"
    
    def generate_insights_report(self, top_n: int = 10) -> str:
        """
        Generate text report of key insights.
        
        Args:
            top_n: Number of top features to include
        
        Returns:
            Formatted text report
        """
        logger.info("Generating insights report...")
        
        explanation = self.get_global_explanation()
        
        if not explanation:
            return "No explanation available"
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': explanation['names'],
            'Importance': explanation['scores']
        })
        
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Generate report
        report = []
        report.append("="*60)
        report.append("MODEL EXPLAINABILITY REPORT")
        report.append("="*60)
        report.append("")
        report.append(f"Top {top_n} Most Important Features:")
        report.append("-"*60)
        
        for idx, row in importance_df.head(top_n).iterrows():
            feature = row['Feature']
            score = row['Importance']
            interpretation = self._interpret_score(score)
            
            report.append(f"\n{idx+1}. {feature}")
            report.append(f"   Importance: {score:.4f}")
            report.append(f"   {interpretation}")
        
        report.append("\n" + "="*60)
        report.append("KEY INSIGHTS:")
        report.append("="*60)
        
        # Categorize features
        top_features = importance_df.head(top_n)
        
        clinical_features = [f for f in top_features['Feature'] if any(x in f for x in ['therapy', 'side_effect', 'diagnosis', 'event'])]
        temporal_features = [f for f in top_features['Feature'] if any(x in f for x in ['day_', 'week_', 'month', 'weekend'])]
        lag_features = [f for f in top_features['Feature'] if 'steps_t_minus' in f]
        rolling_features = [f for f in top_features['Feature'] if 'rolling' in f]
        
        report.append(f"\n• Clinical features in top {top_n}: {len(clinical_features)}")
        if clinical_features:
            report.append(f"  Most important: {clinical_features[0] if clinical_features else 'None'}")
        
        report.append(f"\n• Temporal features in top {top_n}: {len(temporal_features)}")
        if temporal_features:
            report.append(f"  Most important: {temporal_features[0] if temporal_features else 'None'}")
        
        report.append(f"\n• Lag features in top {top_n}: {len(lag_features)}")
        if lag_features:
            report.append(f"  Most important: {lag_features[0] if lag_features else 'None'}")
        
        report.append(f"\n• Rolling features in top {top_n}: {len(rolling_features)}")
        if rolling_features:
            report.append(f"  Most important: {rolling_features[0] if rolling_features else 'None'}")
        
        report.append("\n" + "="*60)
        
        report_text = "\n".join(report)
        
        logger.info("✓ Insights report generated")
        
        return report_text


if __name__ == "__main__":
    print("Explainability module - use with trained models")
