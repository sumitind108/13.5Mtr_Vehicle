

# from flask import Flask, request, render_template
# import pandas as pd
# import matplotlib.pyplot as plt
# import io
# import os
# from werkzeug.utils import secure_filename
# import numpy as np

# app = Flask(__name__)

# def load_data(file_stream, file_name):
#     _, ext = os.path.splitext(file_name)
    
#     if ext == '.xlsx':
#         data = pd.read_excel(file_stream)
#     elif ext == '.ods':
#         data = pd.read_excel(file_stream, engine='odf')
#     else:
#         raise ValueError("Unsupported file type. Please use .xlsx or .ods files.")
    
#     return data

# def plot_parameters(data, parameters):
#     plt.switch_backend('Agg')  # Switch to a non-GUI backend
#     plt.figure(figsize=(12, 8))
    
#     for param in parameters:
#         if param in data.columns:
#             # Optional: Resample or limit data for performance
#             # data_resampled = data.resample('D').mean()  # Example: daily resampling
#             plt.plot(data['Datetime'], data[param], label=param)
#         else:
#             return None  # Exit early if any parameter is invalid
    
#     plt.xlabel('Datetime')
#     plt.ylabel('Values')
#     plt.title('Parameters over Time')
#     plt.legend()
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
    
#     # Save plot to bytes buffer
#     img = io.BytesIO()
#     plt.savefig(img, format='png')
#     img.seek(0)
#     plt.close()
    
#     return img

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     file = request.files['file']
#     parameters = request.form.get('parameters').split(',')
#     parameters = [param.strip() for param in parameters]
    
#     if file:
#         try:
#             data = load_data(file, file.filename)
            
#             # Convert date and time columns to datetime
#             data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], infer_datetime_format=True, errors='coerce')
            
#             img = plot_parameters(data, parameters)
            
#             if img:
#                 # Save plot to disk for download
#                 plot_filename = f'{secure_filename(file.filename)}_plot.png'
#                 img_path = os.path.join('static', plot_filename)
#                 with open(img_path, 'wb') as f:
#                     f.write(img.getvalue())
                
#                 return render_template('results.html', plot_url=plot_filename)
#             else:
#                 return "Parameter(s) not found in the data or datetime parsing failed.", 400
#         except Exception as e:
#             return str(e), 500
    
#     return "No file uploaded.", 400

# if __name__ == '__main__':
#     if not os.path.exists('static'):
#         os.makedirs('static')
#     app.run(port=5003, debug=True)



# ---------------------------------------------------------------------




# from flask import Flask, request, render_template
# import pandas as pd
# import matplotlib.pyplot as plt
# import io
# import os
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# def load_data(file_stream, file_name):
#     _, ext = os.path.splitext(file_name)
    
#     if ext == '.xlsx':
#         data = pd.read_excel(file_stream)
#     elif ext == '.ods':
#         data = pd.read_excel(file_stream, engine='odf')
#     else:
#         raise ValueError("Unsupported file type. Please use .xlsx or .ods files.")
    
#     return data

# def plot_parameters(data, parameters, gain_factors):
#     plt.switch_backend('Agg')  # Switch to a non-GUI backend
#     plt.figure(figsize=(12, 8))

#     # Determine the maximum value across all parameters for setting the y-axis limit
#     max_value = 0
#     for param in parameters:
#         if param in data.columns:
#             max_value = max(max_value, data[param].max())
#         else:
#             return None  # Exit early if any parameter is invalid

#     for param, gain in zip(parameters, gain_factors):
#         if param in data.columns:
#             plt.plot(data['Datetime'], data[param] * gain, label=f"{param} (x{gain})")
    
#     plt.xlabel('Datetime')
#     plt.ylabel('Values')
#     plt.title('Parameters over Time')
#     plt.legend()
#     plt.grid(True)
#     plt.ylim(0, max_value * max(gain_factors))  # Set y-axis limit based on max value and gain factors
#     plt.xticks(rotation=45)
#     plt.tight_layout()
    
#     # Save plot to bytes buffer
#     img = io.BytesIO()
#     plt.savefig(img, format='png')
#     img.seek(0)
#     plt.close()
    
#     return img

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     file = request.files.get('file')
#     parameters = request.form.get('parameters', '').split(',')
#     parameters = [param.strip() for param in parameters]
#     gain_factors = request.form.get('gain_factors', '').split(',')
    
#     try:
#         gain_factors = [float(gain.strip()) for gain in gain_factors]
#     except ValueError:
#         return "Invalid gain factor values provided.", 400
    
#     if file and file.filename:
#         try:
#             data = load_data(file, file.filename)
            
#             # Check for necessary columns
#             required_columns = ['Date', 'Time'] + parameters
#             missing_columns = [col for col in required_columns if col not in data.columns]
#             if missing_columns:
#                 return f"Missing columns: {', '.join(missing_columns)}", 400
            
#             # Convert date and time columns to datetime
#             data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], infer_datetime_format=True, errors='coerce')
            
#             if len(parameters) != len(gain_factors):
#                 return "The number of parameters must match the number of gain factors.", 400
            
#             img = plot_parameters(data, parameters, gain_factors)
            
#             if img:
#                 # Save plot to disk for download
#                 plot_filename = f'{secure_filename(file.filename)}_plot.png'
#                 img_path = os.path.join('static', plot_filename)
#                 with open(img_path, 'wb') as f:
#                     f.write(img.getvalue())
                
#                 return render_template('results.html', plot_url=plot_filename)
#             else:
#                 return "Parameter(s) not found in the data or datetime parsing failed.", 400
#         except Exception as e:
#             return str(e), 500
    
#     return "No file uploaded.", 400

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5007))  # Use environment variable PORT
#     if not os.path.exists('static'):
#         os.makedirs('static')
#     app.run(host='0.0.0.0', port=port, debug=False)



# -----------------------------------------------------------
# from flask import Flask, request, render_template
# import pandas as pd
# import matplotlib.pyplot as plt
# import io
# import os
# import logging
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logging

# def load_data(file_stream, file_name):
#     _, ext = os.path.splitext(file_name)
    
#     if ext == '.xlsx':
#         data = pd.read_excel(file_stream)
#     elif ext == '.ods':
#         data = pd.read_excel(file_stream, engine='odf')
#     else:
#         raise ValueError("Unsupported file type. Please use .xlsx or .ods files.")
    
#     return data

# def plot_parameters(data, parameters, gain_factors):
#     plt.switch_backend('Agg')  # Switch to a non-GUI backend
#     plt.figure(figsize=(8, 5))  # Reduce figure size
#     plt.gcf().set_dpi(60)  # Reduce DPI

#     # Determine the maximum value across all parameters for setting the y-axis limit
#     max_value = 0
#     for param in parameters:
#         if param in data.columns:
#             max_value = max(max_value, data[param].max())
#         else:
#             logging.error(f"Parameter '{param}' not found in data columns.")
#             return None

#     for param, gain in zip(parameters, gain_factors):
#         if param in data.columns:
#             plt.plot(data['Datetime'], data[param] * gain, label=f"{param} (x{gain})")

#     plt.xlabel('Datetime')
#     plt.ylabel('Values')
#     plt.title('Parameters over Time')
#     plt.legend()
#     plt.grid(True)
#     plt.ylim(0, max_value * max(gain_factors))  # Set y-axis limit based on max value and gain factors
#     plt.xticks(rotation=45)
#     plt.tight_layout()
    
#     # Save plot to bytes buffer
#     img = io.BytesIO()
#     plt.savefig(img, format='png', bbox_inches='tight')  # Remove dpi parameter
#     img.seek(0)
#     plt.close()
    
#     return img

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     file = request.files.get('file')
#     parameters = request.form.get('parameters', '').split(',')
#     parameters = [param.strip() for param in parameters]
#     gain_factors = request.form.get('gain_factors', '').split(',')
    
#     try:
#         gain_factors = [float(gain.strip()) for gain in gain_factors]
#     except ValueError:
#         logging.error("Invalid gain factor values provided.")
#         return "Invalid gain factor values provided.", 400
    
#     if file and file.filename:
#         try:
#             logging.info("Loading data...")
#             data = load_data(file, file.filename)
            
#             # Check for necessary columns
#             required_columns = ['Date', 'Time'] + parameters
#             missing_columns = [col for col in required_columns if col not in data.columns]
#             if missing_columns:
#                 logging.error(f"Missing columns: {', '.join(missing_columns)}")
#                 return f"Missing columns: {', '.join(missing_columns)}", 400
            
#             # Convert date and time columns to datetime
#             logging.info("Processing dates...")
#             data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], infer_datetime_format=True, errors='coerce')
            
#             if len(parameters) != len(gain_factors):
#                 logging.error("The number of parameters must match the number of gain factors.")
#                 return "The number of parameters must match the number of gain factors.", 400
            
#             logging.info("Plotting parameters...")
#             img = plot_parameters(data, parameters, gain_factors)
            
#             if img:
#                 # Save plot to disk for download
#                 plot_filename = f'{secure_filename(file.filename)}_plot.png'
#                 img_path = os.path.join('static', plot_filename)
#                 with open(img_path, 'wb') as f:
#                     f.write(img.getvalue())
                
#                 return render_template('results.html', plot_url=plot_filename)
#             else:
#                 logging.error("Plotting failed.")
#                 return "Parameter(s) not found in the data or datetime parsing failed.", 400
#         except Exception as e:
#             logging.error(f"Exception occurred: {e}")
#             return f"An error occurred while processing the file: {e}", 500
    
#     return "No file uploaded.", 400

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5009))  # Use environment variable PORT
#     if not os.path.exists('static'):
#         os.makedirs('static')
#     app.run(host='0.0.0.0', port=port, debug=True)


# -------------------------------

from flask import Flask, request, render_template
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import logging
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logging

def load_data(file_stream, file_name):
    _, ext = os.path.splitext(file_name)
    
    if ext == '.xlsx':
        data = pd.read_excel(file_stream)
    elif ext == '.ods':
        data = pd.read_excel(file_stream, engine='odf')
    else:
        raise ValueError("Unsupported file type. Please use .xlsx or .ods files.")
    
    return data

def plot_parameters(data, parameters, gain_factors):
    plt.switch_backend('Agg')  # Switch to a non-GUI backend
    plt.figure(figsize=(8, 5))  # Reduce figure size
    plt.gcf().set_dpi(60)  # Reduce DPI

    # Determine the maximum value across all parameters for setting the y-axis limit
    max_value = 0
    for param in parameters:
        if param in data.columns:
            max_value = max(max_value, data[param].max())
        else:
            logging.error(f"Parameter '{param}' not found in data columns.")
            return None

    for param, gain in zip(parameters, gain_factors):
        if param in data.columns:
            plt.plot(data['Datetime'], data[param] * gain, label=f"{param} (x{gain})")

    plt.xlabel('Datetime')
    plt.ylabel('Values')
    plt.title('Parameters over Time')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, max_value * max(gain_factors))  # Set y-axis limit based on max value and gain factors
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot to bytes buffer
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')  # Remove dpi parameter
    img.seek(0)
    plt.close()
    
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    parameters = request.form.get('parameters', '').split(',')
    parameters = [param.strip() for param in parameters]
    gain_factors = request.form.get('gain_factors', '').split(',')
    
    try:
        gain_factors = [float(gain.strip()) for gain in gain_factors]
    except ValueError:
        logging.error("Invalid gain factor values provided.")
        return "Invalid gain factor values provided.", 400
    
    if file and file.filename:
        try:
            logging.info("Loading data...")
            data = load_data(file, file.filename)
            
            # Check for necessary columns
            required_columns = ['Date', 'Time'] + parameters
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logging.error(f"Missing columns: {', '.join(missing_columns)}")
                return f"Missing columns: {', '.join(missing_columns)}", 400
            
            # Convert date and time columns to datetime
            logging.info("Processing dates...")
            data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], infer_datetime_format=True, errors='coerce')
            
            if len(parameters) != len(gain_factors):
                logging.error("The number of parameters must match the number of gain factors.")
                return "The number of parameters must match the number of gain factors.", 400
            
            logging.info("Plotting parameters...")
            img = plot_parameters(data, parameters, gain_factors)
            
            if img:
                # Save plot to disk for download
                plot_filename = f'{secure_filename(file.filename)}_plot.png'
                img_path = os.path.join('static', plot_filename)
                with open(img_path, 'wb') as f:
                    f.write(img.getvalue())
                
                return render_template('results.html', plot_url=plot_filename)
            else:
                logging.error("Plotting failed.")
                return "Parameter(s) not found in the data or datetime parsing failed.", 400
        except Exception as e:
            logging.error(f"Exception occurred: {e}")
            return f"An error occurred while processing the file: {e}", 500
    
    return "No file uploaded.", 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5008))  # Use environment variable PORT
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(host='0.0.0.0', port=port, debug=True)
