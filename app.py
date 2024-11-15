from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import json
from werkzeug.utils import secure_filename
from datetime import datetime
from matplotlib.figure import Figure
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from datetime import datetime
import uuid
from DataCleaningRecommender import DataCleaningRecommender


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 16MB max file size
app.secret_key = 'your_secret_key_here'  # Required for flash messages

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'xls'}

def load_dataset(filename):
    """Load dataset from file with proper extension handling"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_ext = filename.rsplit('.', 1)[1].lower()
    
    try:
        if file_ext == 'csv':
            return pd.read_csv(file_path)
        elif file_ext in ['xlsx', 'xls']:
            return pd.read_excel(file_path)
    except Exception as e:
        flash(f"Error loading dataset: {str(e)}", 'error')
        return None

def get_dataset_summary(df):
    """Generate summary statistics for the dataset"""
    return {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }

def detect_and_convert_dates(df):
    """Detect and convert potential date columns to datetime"""
    for column in df.select_dtypes(['object']):
        try:
            # Try converting to datetime
            pd.to_datetime(df[column], errors='raise')
            df[column] = pd.to_datetime(df[column])
        except (ValueError, TypeError):
            continue
    return df

def standardize_column_names(df):
    """Standardize column names: lowercase, replace spaces with underscore"""
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^\w\s]', '')
    return df

def remove_constant_columns(df):
    """Remove columns that have constant values"""
    constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
    return df.drop(columns=constant_columns), constant_columns

def convert_to_numeric(df):
    """Convert string columns to numeric where possible"""
    converted_columns = []
    for column in df.select_dtypes(['object']):
        try:
            # Remove currency symbols and commas
            temp_col = df[column].str.replace('[$,]', '', regex=True)
            # Convert to numeric
            df[column] = pd.to_numeric(temp_col)
            converted_columns.append(column)
        except (ValueError, TypeError):
            continue
    return df, converted_columns

def handle_special_characters(df):
    """Remove or replace special characters in string columns"""
    for column in df.select_dtypes(['object']):
        df[column] = df[column].str.replace('[^\w\s]', '', regex=True)
    return df

def remove_high_correlation(df, threshold=0.95):
    """Remove highly correlated numerical columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return df, []
    
    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    return df.drop(columns=to_drop), to_drop

def create_line_plot(df, x_column, y_column):
    """Create a line plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_column], df[y_column], marker='o')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'{y_column} over {x_column}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot with unique identifier
    plot_filename = f'line_plot_{uuid.uuid4().hex[:8]}.png'
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], plot_filename))
    plt.close()
    return plot_filename

def create_bar_plot(df, x_column, y_column):
    """Create a bar plot"""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=x_column, y=y_column)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'{y_column} by {x_column}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_filename = f'bar_plot_{uuid.uuid4().hex[:8]}.png'
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], plot_filename))
    plt.close()
    return plot_filename

def create_scatter_plot(df, x_column, y_column):
    """Create a scatter plot"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_column, y=y_column)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'{y_column} vs {x_column}')
    plt.tight_layout()
    
    plot_filename = f'scatter_plot_{uuid.uuid4().hex[:8]}.png'
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], plot_filename))
    plt.close()
    return plot_filename

def create_heatmap(df, columns):
    """Create a correlation heatmap"""
    plt.figure(figsize=(12, 8))
    correlation = df[columns].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    
    plot_filename = f'heatmap_{uuid.uuid4().hex[:8]}.png'
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], plot_filename))
    plt.close()
    return plot_filename

def create_distribution_plot(df, column):
    """Create a distribution plot"""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, kde=True)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
    
    plot_filename = f'dist_plot_{uuid.uuid4().hex[:8]}.png'
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], plot_filename))
    plt.close()
    return plot_filename

def create_time_series_plot(df, date_column, value_column):
    """Create a time series plot"""
    plt.figure(figsize=(12, 6))
    df[date_column] = pd.to_datetime(df[date_column])
    plt.plot(df[date_column], df[value_column], marker='o')
    plt.xlabel(date_column)
    plt.ylabel(value_column)
    plt.title(f'{value_column} Time Series')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    
    plot_filename = f'timeseries_{uuid.uuid4().hex[:8]}.png'
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], plot_filename))
    plt.close()
    return plot_filename
    


@app.route('/visualization/<filename>', methods=['GET', 'POST'])
def visualization_page(filename):
    df = load_dataset(filename)
    if df is None:
        flash('Could not load dataset', 'error')
        return redirect(url_for('index'))
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    if request.method == 'POST':
        try:
            viz_type = request.form.get('visualization_type')
            plot_filename = None
            
            if viz_type == 'line':
                x_col = request.form.get('x_column')
                y_col = request.form.get('y_column')
                plt.figure(figsize=(12, 6))
                plt.plot(df[x_col], df[y_col], marker='o')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f'{y_col} vs {x_col} Line Plot')
                plt.xticks(rotation=45)
                
            elif viz_type == 'bar':
                x_col = request.form.get('x_column')
                y_col = request.form.get('y_column')
                plt.figure(figsize=(12, 6))
                sns.barplot(data=df, x=x_col, y=y_col)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f'{y_col} by {x_col} Bar Plot')
                plt.xticks(rotation=45)
                
            elif viz_type == 'scatter':
                x_col = request.form.get('x_column')
                y_col = request.form.get('y_column')
                color_col = request.form.get('color_column')
                plt.figure(figsize=(12, 6))
                if color_col:
                    plt.scatter(df[x_col], df[y_col], c=df[color_col])
                    plt.colorbar(label=color_col)
                else:
                    plt.scatter(df[x_col], df[y_col])
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f'{y_col} vs {x_col} Scatter Plot')
                
            elif viz_type == 'histogram':
                col = request.form.get('column')
                bins = int(request.form.get('bins', 30))
                plt.figure(figsize=(12, 6))
                plt.hist(df[col], bins=bins)
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.title(f'Histogram of {col}')
                
            elif viz_type == 'box':
                cols = request.form.getlist('columns')
                plt.figure(figsize=(12, 6))
                df[cols].boxplot()
                plt.xticks(rotation=45)
                plt.ylabel('Value')
                plt.title('Box Plot')
                
            elif viz_type == 'violin':
                x_col = request.form.get('x_column')
                y_col = request.form.get('y_column')
                plt.figure(figsize=(12, 6))
                sns.violinplot(data=df, x=x_col, y=y_col)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f'Violin Plot of {y_col} by {x_col}')
                plt.xticks(rotation=45)
                
            elif viz_type == 'heatmap':
                cols = request.form.getlist('columns')
                plt.figure(figsize=(12, 8))
                sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm')
                plt.title('Correlation Heatmap')
                
            elif viz_type == 'pie':
                col = request.form.get('column')
                plt.figure(figsize=(10, 10))
                df[col].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.title(f'Pie Chart of {col}')
                
            elif viz_type == 'density':
                col = request.form.get('column')
                plt.figure(figsize=(12, 6))
                sns.kdeplot(data=df[col], fill=True)
                plt.xlabel(col)
                plt.ylabel('Density')
                plt.title(f'Density Plot of {col}')
            
            plt.tight_layout()
            plot_filename = f'viz_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], plot_filename))
            plt.close()
            
            return render_template('visualization_page.html',
                               filename=filename,
                               plot_image=plot_filename,
                               columns=df.columns.tolist(),
                               numeric_columns=numeric_columns,
                               categorical_columns=categorical_columns,
                               datetime_columns=datetime_columns)
            
        except Exception as e:
            flash(f'Error creating visualization: {str(e)}', 'error')
    
    return render_template('visualization_page.html',
                       filename=filename,
                       columns=df.columns.tolist(),
                       numeric_columns=numeric_columns,
                       categorical_columns=categorical_columns,
                       datetime_columns=datetime_columns)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            # Get file size before saving
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)  # Reset file pointer
            
            # Convert size to MB for display
            size_mb = file_size / (1024 * 1024)
            
            # Check if file size is within new limit
            if size_mb > 100:
                flash(f'File size ({size_mb:.1f}MB) exceeds the 100MB limit', 'error')
                return redirect(url_for('index'))
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # For large CSV files, use chunking to read the file
            if filename.endswith('.csv'):
                # Save file first
                file.save(file_path)
                
                # Try to read the first few rows to verify file integrity
                try:
                    chunk_size = 1000  # Adjust based on your needs
                    df_chunk = pd.read_csv(file_path, nrows=chunk_size)
                    # If successful, flash a message about the file size
                    flash(f'Large file ({size_mb:.1f}MB) uploaded successfully. Processing may take longer.', 'warning')
                except Exception as e:
                    os.remove(file_path)  # Clean up the file if there's an error
                    flash(f'Error reading file: {str(e)}', 'error')
                    return redirect(url_for('index'))
                
            else:  # For Excel files
                file.save(file_path)
                flash(f'File ({size_mb:.1f}MB) uploaded successfully', 'success')
            
            return redirect(url_for('data_cleaning', filename=filename))
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload CSV or Excel file.', 'error')
        return redirect(url_for('index'))

# Add error handler for large files
@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File too large. Maximum file size is 100MB.', 'error')
    return redirect(url_for('index')), 413

@app.route('/advanced_clean/<filename>/<operation>', methods=['POST'])
def advanced_clean(filename, operation):
    df = load_dataset(filename)
    if df is None:
        flash('Could not load dataset', 'error')
        return redirect(url_for('index'))
    
    try:
        result = ''
        if operation == 'detect_dates':
            df = detect_and_convert_dates(df)
            result = "Date columns detected and converted to datetime format"
            
            
        elif operation == 'remove_constant':
            df, constant_cols = remove_constant_columns(df)
            result = f"Removed {len(constant_cols)} constant columns: {', '.join(constant_cols)}"
            
        elif operation == 'convert_numeric':
            df, converted_cols = convert_to_numeric(df)
            result = f"Converted {len(converted_cols)} columns to numeric: {', '.join(converted_cols)}"
            
        elif operation == 'handle_special_chars':
            df = handle_special_characters(df)
            result = "Special characters removed from string columns"
            
        elif operation == 'remove_correlation':
            threshold = float(request.form.get('threshold', 0.95))
            df, dropped_cols = remove_high_correlation(df, threshold)
            result = f"Removed {len(dropped_cols)} highly correlated columns: {', '.join(dropped_cols)}"
        
        # Save the cleaned dataset
        df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename), index=False)
        flash(result, 'success')
        
        return redirect(url_for('data_cleaning', filename=filename))
        
    except Exception as e:
        flash(f'Error during advanced cleaning: {str(e)}', 'error')
        return redirect(url_for('data_cleaning', filename=filename))


@app.route('/data_cleaning/<filename>')
def data_cleaning(filename):
    df = load_dataset(filename)
    if df is None:
        return redirect(url_for('index'))
    
    result = ''
    
    summary = get_dataset_summary(df)
    
    # Add profile information
    profile = {
        'basic_info': {
            'rows': df.shape[0],
            'columns': df.shape[1],
            'missing_cells': df.isnull().sum().sum(),
            'missing_cells_pct': round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2),
            'duplicate_rows': df.duplicated().sum()
        },
        'column_types': {
            'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical': df.select_dtypes(include=['object']).columns.tolist(),
            'datetime': df.select_dtypes(include=['datetime64']).columns.tolist()
        }
    }
    
    # Calculate quality score (simple example)
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (df.isnull().sum().sum() / total_cells) * 100
    duplicate_pct = (df.duplicated().sum() / df.shape[0]) * 100
    quality_score = round(100 - (missing_pct + duplicate_pct) / 2)
    
    quality_deductions = []
    if missing_pct > 5:
        quality_deductions.append(f"High missing values: {missing_pct:.1f}%")
    if duplicate_pct > 1:
        quality_deductions.append(f"Duplicate rows: {duplicate_pct:.1f}%")

    # Get cleaning recommendations
    recommender = DataCleaningRecommender()
    recommendations, recommendation_score = recommender.analyze_dataset(df)

    return render_template('data_cleaning.html', 
                       result=result, 
                       filename=filename, 
                       summary=get_dataset_summary(df),
                       profile=profile,
                       quality_score=quality_score,
                       quality_deductions=quality_deductions,
                       recommendations=recommendations,
                       recommendation_score=recommendation_score)

@app.route('/perform_operation/<filename>/<operation>', methods=['GET', 'POST'])
def perform_operation(filename, operation):
    df = load_dataset(filename)
    if df is None:
        return redirect(url_for('index'))
    
    result = ''
    try:
        if operation == 'smart_clean':
            # Store original metrics BEFORE any operations
            original_rows = len(df)
            original_missing = df.isnull().sum().sum()
            
            # Create a copy of the dataframe to work with
            df_clean = df.copy()
            
            # Remove columns with too many missing values (>50%)
            missing_pct = df_clean.isnull().sum() / len(df_clean)
            df_clean = df_clean.drop(columns=missing_pct[missing_pct > 0.5].index)
            
            # Fill numeric columns with median
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
            # Fill categorical columns with mode
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
            
            # Remove duplicate rows
            df_clean = df_clean.drop_duplicates()
            
            # Handle outliers for numeric columns
            for col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                df_clean[col] = df_clean[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
            
            # Save cleaned dataset and update df reference
            df = df_clean
            df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename), index=False)
            
            # Generate report
            cleaned_rows = len(df)
            cleaned_missing = df.isnull().sum().sum()
            result = f"""
            Cleaning Complete:
            - Rows: {original_rows} → {cleaned_rows} ({original_rows - cleaned_rows} removed)
            - Missing Values: {original_missing} → {cleaned_missing}
            - Outliers handled in {len(numeric_cols)} numeric columns
            """

        elif operation == 'missing_data_per_column':
            missing_data = df.isnull().sum()
            missing_percentages = (missing_data / len(df) * 100).round(2)
            result = pd.DataFrame({
                'Missing Values': missing_data,
                'Percentage': missing_percentages
            }).to_html(classes='table table-striped')
    
        elif operation == 'total_missing_values':
            total_missing = df.isnull().sum().sum()
            missing_percentage = (total_missing / (df.shape[0] * df.shape[1]) * 100).round(2)
            result = f"Total Missing Values: {total_missing} ({missing_percentage}% of all data)"

        elif operation == 'percent_missing':
            percent_missing = (df.isnull().sum() / len(df) * 100).round(2)
            result = percent_missing.to_frame('Percent Missing').to_html(classes='table table-striped')

        elif operation == 'drop_columns_with_missing':
            original_cols = df.shape[1]
            df = df.dropna(axis=1)
            dropped_cols = original_cols - df.shape[1]
            df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename), index=False)
            result = f"Dropped {dropped_cols} columns with missing values. {df.shape[1]} columns remaining."

        elif operation == 'fillna_with_value':
            fill_value = request.form.get('fill_value')
            if fill_value is not None:
                df.fillna(float(fill_value), inplace=True)
                df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename), index=False)
                result = f"Missing values filled with {fill_value}"

        elif operation == 'fillna_bfill_ffill':
            df.fillna(method='ffill', inplace=True)
            df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename), index=False)
            result = "Missing values filled using forward fill"
        
        elif operation == 'bfill':
            df.fillna(method='bfill', inplace=True)
            df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename), index=False)
            result = "Missing values filled using backward fill"


        elif operation == 'remove_duplicates':
            original_rows = len(df)
            df.drop_duplicates(inplace=True)
            dropped_rows = original_rows - len(df)
            df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename), index=False)
            result = f"Removed {dropped_rows} duplicate rows. {len(df)} rows remaining."

        elif operation == 'handle_outliers':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
            df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename), index=False)
            result = "Outliers handled using IQR method"

        elif operation == 'visualize':
            plot_type = request.form.get('plot_type')
            columns = request.form.getlist('columns')
            
            plt.figure(figsize=(10, 6))
            if plot_type == 'box_plot':
                df[columns].boxplot()
                plt.title('Box Plot')
            elif plot_type == 'heatmap':
                sns.heatmap(df[columns].corr(), annot=True, cmap='coolwarm')
                plt.title('Correlation Heatmap')
            elif plot_type == 'histogram':
                df[columns[0]].hist()
                plt.title(f'Histogram of {columns[0]}')
            elif plot_type == 'scatter':
                if len(columns) >= 2:
                    plt.scatter(df[columns[0]], df[columns[1]])
                    plt.xlabel(columns[0])
                    plt.ylabel(columns[1])
                    plt.title(f'Scatter Plot: {columns[0]} vs {columns[1]}')
        
        elif operation == 'encode_categorical':
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[f"{col}_encoded"] = pd.factorize(df[col])[0]
            df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename), index=False)
            result = f"Encoded {len(categorical_cols)} categorical columns"
            
        elif operation == 'normalize_numeric':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
            df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename), index=False)
            result = f"Normalized {len(numeric_cols)} numeric columns"

        elif operation == 'standardize_names':
            df = standardize_column_names(df)
            result = "Column names standardized"
            
            plot_filename = f'plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], plot_filename))
            plt.close()
            return render_template('visualizations.html', 
                                plot_image=plot_filename, 
                                filename=filename, 
                                columns=df.columns.tolist())

        # Calculate profile information AFTER operations
        profile = {
            'basic_info': {
                'rows': df.shape[0],
                'columns': df.shape[1],
                'missing_cells': df.isnull().sum().sum(),
                'missing_cells_pct': round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2),
                'duplicate_rows': df.duplicated().sum()
            },
            'column_types': {
                'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical': df.select_dtypes(include=['object']).columns.tolist(),
                'datetime': df.select_dtypes(include=['datetime64']).columns.tolist()
            }
        }

        # Calculate quality metrics
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = (df.isnull().sum().sum() / total_cells) * 100
        duplicate_pct = (df.duplicated().sum() / df.shape[0]) * 100
        quality_score = round(100 - (missing_pct + duplicate_pct) / 2)
        
        quality_deductions = []
        if missing_pct > 5:
            quality_deductions.append(f"High missing values: {missing_pct:.1f}%")
        if duplicate_pct > 1:
            quality_deductions.append(f"Duplicate rows: {duplicate_pct:.1f}%")

        return render_template('data_cleaning.html', 
                           result=result, 
                           filename=filename, 
                           summary=get_dataset_summary(df),
                           profile=profile,
                           quality_score=quality_score,
                           quality_deductions=quality_deductions)

    except Exception as e:
        flash(f"Error performing operation: {str(e)}", 'error')
        
        # Even in case of error, we need to pass all required variables
        profile = {
            'basic_info': {
                'rows': df.shape[0],
                'columns': df.shape[1],
                'missing_cells': df.isnull().sum().sum(),
                'missing_cells_pct': 0,
                'duplicate_rows': 0
            },
            'column_types': {
                'numeric': [],
                'categorical': [],
                'datetime': []
            }
        }
        
        return render_template('data_cleaning.html', 
                           result=result, 
                           filename=filename, 
                           summary=get_dataset_summary(df),
                           profile=profile,
                           quality_score=0,
                           quality_deductions=[])
    
@app.route('/export_data/<filename>', methods=['POST'])
def export_data(filename):
    df = load_dataset(filename)
    if df is None:
        return redirect(url_for('index'))
    
    export_format = request.form.get('export_format', 'csv')
    
    try:
        if export_format == 'csv':
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'cleaned_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
        elif export_format == 'excel':
            output = io.BytesIO()
            df.to_excel(output, index=False)
            output.seek(0)
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name=f'cleaned_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
            )
        elif export_format == 'json':
            return send_file(
                io.BytesIO(df.to_json(orient='records').encode('utf-8')),
                mimetype='application/json',
                as_attachment=True,
                download_name=f'cleaned_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
    except Exception as e:
        flash(f"Error exporting data: {str(e)}", 'error')
        return redirect(url_for('data_cleaning', filename=filename))

@app.route('/preview_data/<filename>')
def preview_data(filename):
    df = load_dataset(filename)
    if df is None:
        return jsonify({'error': 'Could not load dataset'})
    
    return jsonify({
        'head': df.head().to_dict(orient='records'),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'shape': df.shape
    })

@app.route('/download_plot/<filename>')
def download_plot(filename):
    try:
        return send_file(
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        flash(f"Error downloading plot: {str(e)}", 'error')
        return redirect(url_for('index'))
    
# Add this route to your existing app.py file, before the if __name__ == '__main__': line
@app.route('/smart_clean/<filename>', methods=['POST'])
def smart_clean(filename):
    """Automatically clean the dataset using best practices"""
    df = load_dataset(filename)
    if df is None:
        flash('Could not load dataset', 'error')
        return redirect(url_for('index'))
    
    try:
        # Store original metrics for comparison
        original_rows = len(df)
        original_missing = df.isnull().sum().sum()
        original_cols = len(df.columns)
        
        # Remove columns with too many missing values (>50%)
        missing_pct = df.isnull().sum() / len(df)
        high_missing_cols = missing_pct[missing_pct > 0.5].index
        df = df.drop(columns=high_missing_cols)
        
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Remove duplicate rows
        original_dupes = df.duplicated().sum()
        df = df.drop_duplicates()
        
        # Handle outliers for numeric columns using IQR method
        outliers_handled = 0
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
            if outlier_mask.any():
                df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
                outliers_handled += outlier_mask.sum()
        
        # Generate detailed cleaning report
        cleaned_rows = len(df)
        cleaned_missing = df.isnull().sum().sum()
        
        cleaning_report = (
            f"Smart Cleaning Complete:\n"
            f"• Rows: {original_rows:,} → {cleaned_rows:,} ({original_rows - cleaned_rows:,} removed)\n"
            f"• Columns: {original_cols} → {len(df.columns)} ({len(high_missing_cols)} high-missing columns removed)\n"
            f"• Missing Values: {original_missing:,} → {cleaned_missing:,}\n"
            f"• Duplicate Rows Removed: {original_dupes:,}\n"
            f"• Outliers Handled: {outliers_handled:,} values in {len(numeric_cols)} numeric columns"
        )
        
        # Save cleaned dataset
        cleaned_filename = f'cleaned_{filename}'
        df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename), index=False)
        
        flash(cleaning_report, 'success')
        return redirect(url_for('data_cleaning', filename=cleaned_filename))
        
    except Exception as e:
        flash(f'Error during smart cleaning: {str(e)}', 'error')
        return redirect(url_for('data_cleaning', filename=filename))

@app.route('/get_recommendations/<filename>')
def get_recommendations(filename):
    df = load_dataset(filename)
    if df is None:
        return jsonify({'error': 'Could not load dataset'})
    
    recommender = DataCleaningRecommender()
    recommendations, total_score = recommender.analyze_dataset(df)
    
    return jsonify({
        'recommendations': recommendations,
        'total_score': total_score
    })


if __name__ == '__main__':
    app.run(debug=True)