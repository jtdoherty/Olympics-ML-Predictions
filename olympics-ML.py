#inspired by Dataquest
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt  # Added missing import
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

def load_and_clean_data(file_path: str, required_columns: List[str]) -> pd.DataFrame:
    """
    Load and clean the Olympics dataset.
    
    Args:
        file_path: Path to the CSV file
        required_columns: List of columns to keep
    
    Returns:
        Cleaned DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        df = df[required_columns].copy()
        df["medals"] = pd.to_numeric(df["medals"], errors='coerce')
        cleaned_df = df.dropna()
        
        if cleaned_df.empty:
            raise ValueError("After cleaning, the DataFrame is empty")
            
        return cleaned_df
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file at {file_path}")
    except Exception as e:
        raise Exception(f"Error processing data: {str(e)}")

def create_visualizations(df: pd.DataFrame, save_plots: bool = False) -> None:
    """
    Create exploratory visualizations.
    
    Args:
        df: Input DataFrame
        save_plots: Whether to save plots to disk
    """
    try:
        # Set style for better looking plots
        plt.style.use('default')  # Changed from 'seaborn' to 'default'
        
        # Create figure with subplots for better organization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Olympics Medal Analysis', fontsize=14, y=1.05)
        
        # Color palette
        main_color = '#1f77b4'  # A nice blue color
        
        # Athletes vs Medals
        sns.regplot(x='athletes', y='medals', data=df, ax=ax1, 
                   scatter_kws={'alpha':0.4, 'color': main_color},
                   line_kws={'color': 'red'})
        ax1.set_title('Athletes vs Medals')
        ax1.set_xlabel('Number of Athletes')
        ax1.set_ylabel('Number of Medals')
        ax1.grid(True, alpha=0.3)
        
        # Age vs Medals
        sns.regplot(x='age', y='medals', data=df, ax=ax2,
                   scatter_kws={'alpha':0.4, 'color': main_color},
                   line_kws={'color': 'red'})
        ax2.set_title('Age vs Medals')
        ax2.set_xlabel('Average Age')
        ax2.set_ylabel('Number of Medals')
        ax2.grid(True, alpha=0.3)
        
        # Medals distribution
        df["medals"].hist(ax=ax3, bins=30, color=main_color, alpha=0.7)
        ax3.set_title('Medals Distribution')
        ax3.set_xlabel('Number of Medals')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_plots:
            try:
                plt.savefig("C:\Git\Olympics-ML-Predictions\olympics_analysis.png", dpi=300, bbox_inches='tight')
                print("Plot saved successfully as 'olympics_analysis.png'")
            except Exception as e:
                print(f"Warning: Could not save plot: {str(e)}")
        
        plt.show()
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
    finally:
        plt.close()


def split_data(df: pd.DataFrame, split_year: int = 2016) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets based on year.
    
    Args:
        df: Input DataFrame
        split_year: Year to split the data
    
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = df[df["year"] < split_year].copy()
    test_df = df[df["year"] >= split_year].copy()
    
    if train_df.empty or test_df.empty:
        raise ValueError(f"Invalid split: One of the datasets is empty. Split year: {split_year}")
        
    return train_df, test_df

def train_model(train_df: pd.DataFrame, predictors: List[str], target: str) -> Tuple[LinearRegression, StandardScaler]:
    """
    Train the model with standardized features.
    
    Args:
        train_df: Training DataFrame
        predictors: List of predictor columns
        target: Target column name
    
    Returns:
        Tuple of (trained_model, scaler)
    """
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_df[predictors])
        
        model = LinearRegression()
        model.fit(X_train_scaled, train_df[target])
        
        # Print model coefficients for interpretation
        for predictor, coef in zip(predictors, model.coef_):
            print(f"Coefficient for {predictor}: {coef:.4f}")
        print(f"Intercept: {model.intercept_:.4f}")
        
        return model, scaler
    except Exception as e:
        raise Exception(f"Error training model: {str(e)}")

def make_predictions(model: LinearRegression, 
                    test_df: pd.DataFrame, 
                    predictors: List[str], 
                    scaler: StandardScaler) -> np.ndarray:
    """
    Make predictions on test data.
    
    Args:
        model: Trained model
        test_df: Test DataFrame
        predictors: List of predictor columns
        scaler: Fitted StandardScaler
    
    Returns:
        Array of predictions
    """
    try:
        X_test_scaled = scaler.transform(test_df[predictors])
        predictions = model.predict(X_test_scaled)
        return np.maximum(predictions.round(), 0)  # Ensure non-negative predictions
    except Exception as e:
        raise Exception(f"Error making predictions: {str(e)}")

def evaluate_model(test_df: pd.DataFrame, 
                  predictions: np.ndarray, 
                  target: str = "medals") -> dict:
    """
    Evaluate model performance.
    
    Args:
        test_df: Test DataFrame
        predictions: Model predictions
        target: Target column name
    
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        actual = test_df[target]
        metrics = {
            'MAE': mean_absolute_error(actual, predictions),
            'R2': r2_score(actual, predictions),
            'Mean Error Ratio': np.mean(np.abs(actual - predictions) / np.maximum(actual, 1))  # Avoid division by zero
        }
        return metrics
    except Exception as e:
        raise Exception(f"Error evaluating model: {str(e)}")

def analyze_team_performance(test_df: pd.DataFrame, 
                           predictions: np.ndarray, 
                           teams_of_interest: List[str]) -> pd.DataFrame:
    """
    Analyze performance for specific teams.
    
    Args:
        test_df: Test DataFrame
        predictions: Model predictions
        teams_of_interest: List of team names to analyze
    
    Returns:
        DataFrame with team performance metrics
    """
    try:
        test_df = test_df.copy()
        test_df['predictions'] = predictions
        test_df['abs_error'] = np.abs(test_df['medals'] - predictions)
        
        team_analysis = []
        for team in teams_of_interest:
            team_data = test_df[test_df['team'] == team]
            if not team_data.empty:
                team_analysis.append({
                    'team': team,
                    'actual_medals_mean': team_data['medals'].mean(),
                    'predicted_medals_mean': team_data['predictions'].mean(),
                    'mae': team_data['abs_error'].mean()
                })
            else:
                print(f"Warning: No data found for team {team}")
        
        return pd.DataFrame(team_analysis)
    except Exception as e:
        raise Exception(f"Error analyzing team performance: {str(e)}")

def main():
    # Configuration
    FILE_PATH = r"Olympics-ML-Predictions\teams.csv"
    REQUIRED_COLUMNS = ["team", "country", "year", "athletes", "age", "prev_medals", "medals"]
    PREDICTORS = ["athletes", "prev_medals"]
    TARGET = "medals"
    TEAMS_OF_INTEREST = ["USA", "IND"]
    
    try:
        # Load and process data
        print("Loading and processing data...")
        df = load_and_clean_data(FILE_PATH, REQUIRED_COLUMNS)
        
        print("\nCreating visualizations...")
        create_visualizations(df, save_plots=True)
        
        # Split data and train model
        print("\nSplitting data and training model...")
        train_df, test_df = split_data(df)
        model, scaler = train_model(train_df, PREDICTORS, TARGET)
        
        # Make predictions and evaluate
        print("\nMaking predictions and evaluating model...")
        predictions = make_predictions(model, test_df, PREDICTORS, scaler)
        metrics = evaluate_model(test_df, predictions)
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Analyze specific teams
        print("\nAnalyzing team performance...")
        team_analysis = analyze_team_performance(test_df, predictions, TEAMS_OF_INTEREST)
        print("\nTeam Performance Analysis:")
        print(team_analysis.to_string())
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
