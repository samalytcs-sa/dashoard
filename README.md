# Arabic Hashtag & Sentiment Analysis Dashboard

A comprehensive Flask-based dashboard for analyzing hashtags and performing sentiment analysis on Arabic and English text data from Excel files.

## Features

### 🔍 **Hashtag Analysis**
- Extract hashtags from Arabic and English text
- Generate interactive bar charts showing top 20 hashtags
- Create beautiful word clouds for hashtag visualization
- Support for Arabic Unicode characters in hashtags

### 💭 **Advanced Sentiment Analysis**
- **Arabic Sentiment Analysis**: Uses Hugging Face's `Ammar-alhaj-ali/arabic-MARBERT-sentiment` model
- **Multilingual Support**: Handles both Arabic and English text
- **Fallback System**: Basic sentiment analysis when advanced models aren't available
- **Interactive Visualizations**: Pie charts showing sentiment distribution

### 📊 **Data Visualization**
- Modern, responsive web interface
- Interactive Plotly charts
- Real-time data updates
- Beautiful gradient design with hover effects

### 📈 **Dashboard Features**
- Data overview with file statistics
- Model status monitoring
- Responsive design for all devices
- Real-time refresh capabilities

## Installation

1. **Clone or download the project files**

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your Excel files in the project directory:**
   - `ConversationStreamDistribution_0606971e-846a-48bd-895e-8dbf89f18838_1.xlsx`
   - `ConversationStreamDistribution_0606971e-846a-48bd-895e-8dbf89f18838_2.xlsx`

## Usage

1. **Start the Flask application:**
   ```bash
   python app.py
   ```

2. **Open your web browser and navigate to:**
   ```
   http://localhost:5000
   ```

3. **The dashboard will automatically:**
   - Load the Arabic sentiment analysis model
   - Process your Excel files
   - Generate interactive visualizations

## Arabic Sentiment Analysis Model

This dashboard uses the **Arabic MARBERT sentiment analysis model** from Hugging Face:

- **Model**: `Ammar-alhaj-ali/arabic-MARBERT-sentiment`
- **Capabilities**: Analyzes Arabic text for positive, negative, and neutral sentiments
- **Training**: Fine-tuned on KAUST dataset with 3 labels
- **Performance**: High accuracy for Arabic sentiment classification

### Model Usage Example:
```python
from transformers import pipeline

# Initialize the pipeline
sentiment_analyzer = pipeline('text-classification', 
                            model='Ammar-alhaj-ali/arabic-MARBERT-sentiment')

# Analyze Arabic text
result = sentiment_analyzer('لقد استمتعت بالحفلة')
# Output: [{'label': 'positive', 'score': 0.9577557444572449}]
```

## File Structure

```
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Dashboard HTML template
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── *.xlsx               # Your Excel data files
```

## API Endpoints

- `GET /` - Main dashboard page
- `GET /api/hashtag-analysis` - Hashtag analysis data
- `GET /api/sentiment-analysis` - Sentiment analysis results
- `GET /api/data-overview` - Data statistics overview
- `GET /api/model-status` - Arabic model loading status

## Supported Data Columns

The dashboard automatically searches for text content in these columns:
- `Message`
- `Content` 
- `Text`
- `Description`
- `Body`
- `Comment`

## Features in Detail

### Hashtag Extraction
- Uses enhanced regex pattern: `#[\w\u0600-\u06FF]+`
- Supports Arabic Unicode range (U+0600 to U+06FF)
- Extracts hashtags from multiple text columns
- Counts frequency and shows top 20 hashtags

### Sentiment Analysis
- **Primary**: Arabic MARBERT model for accurate Arabic sentiment analysis
- **Fallback**: Basic keyword-based analysis for both Arabic and English
- **Arabic Keywords**: Includes common positive/negative Arabic words
- **Performance**: Processes up to 512 characters per text sample

### Visualizations
- **Bar Charts**: Interactive hashtag frequency charts
- **Pie Charts**: Sentiment distribution with custom colors
- **Word Clouds**: Beautiful hashtag visualizations
- **Statistics Cards**: Key metrics display

## Technical Requirements

- Python 3.7+
- Flask 2.3+
- Transformers 4.56+
- PyTorch 2.2+
- Pandas 2.3+
- Plotly 6.3+

## Performance Notes

- The dashboard samples the first 100 entries per text column for faster processing
- You can modify this in the `sentiment_analysis()` function
- Arabic model loading may take a few minutes on first run
- Word cloud generation supports up to 100 words

## Troubleshooting

### Model Loading Issues
If the Arabic model fails to load:
- Check internet connection (model downloads from Hugging Face)
- Ensure sufficient disk space (model is ~400MB)
- The dashboard will fall back to basic sentiment analysis

### Excel File Issues
- Ensure Excel files are in the correct format
- Check file names match exactly
- Verify files are not corrupted or password-protected

### Memory Issues
- Reduce sample size in sentiment analysis
- Close other applications if running low on RAM
- Consider processing files separately for very large datasets

## Contributing

Feel free to enhance this dashboard by:
- Adding more Arabic sentiment models
- Improving hashtag extraction patterns
- Adding more visualization types
- Supporting additional file formats

## License

This project is open source and available under the MIT License.# dashoard
