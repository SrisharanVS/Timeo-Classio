# Time Series Predictor Web Interface

A modern web interface for the Time Series Predictor application, built with FastAPI and a responsive UI.

## Features

- Upload CSV files containing time series data
- Automatic model selection based on time series characteristics
- Beautiful visualization of time series and predictions
- Responsive design that works on all devices
- Real-time feedback during processing

## Requirements

- Python 3.8+
- FastAPI
- Uvicorn
- Pandas
- NumPy
- Other dependencies from the main project

## Installation

1. Install the required dependencies:

```bash
pip install fastapi uvicorn python-multipart jinja2 aiofiles
```

2. Make sure you have the trained models in the `models` directory.

## Running the Application

1. Navigate to the web directory:

```bash
cd src/web
```

2. Start the FastAPI server:

```bash
uvicorn app:app --reload
```

3. Open your browser and go to `http://localhost:8000`

## API Documentation

FastAPI automatically generates API documentation. You can access it at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## File Format

The application expects CSV files with a `point_value` column containing the time series data.

Example:
```
point_timestamp,point_value
2019-07-14,6
2019-07-15,7
2019-07-16,6
...
```

## Troubleshooting

- If you see an error about missing models, make sure you have trained the models first.
- If the chart doesn't display correctly, check that your browser supports Canvas and JavaScript.
- For large files, the upload might take a few seconds. The loading spinner will indicate progress. 