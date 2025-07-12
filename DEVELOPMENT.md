# AI-based Lithography Hotspot Detection - Development Setup

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional, for version control)

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py
   ```

3. **Access the Application**
   - Open your browser and navigate to `http://localhost:8501`
   - The application will automatically reload when you make changes

## Development

### Project Structure
```
├── app.py                 # Main Streamlit application
├── models/               # AI model implementations
│   ├── cyclegan.py      # CycleGAN domain adaptation
│   ├── classifier.py    # Hotspot classification models
│   └── gradcam.py       # Grad-CAM visualization
├── utils/               # Utility functions
│   ├── image_processing.py
│   ├── model_utils.py
│   └── ui_components.py
├── assets/              # Static assets and styling
├── config.py           # Application configuration
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### VS Code Tasks

The project includes VS Code tasks for common operations:

- **Run Streamlit App**: Launch the application in development mode
- **Install Dependencies**: Install Python packages from requirements.txt
- **Format Code**: Format Python code using Black
- **Lint Code**: Check code quality with Flake8

### Configuration

Application settings can be modified in `config.py`:

- Model configurations
- UI settings  
- Performance metrics
- Image processing parameters

### Adding New Models

1. Implement your model class in the `models/` directory
2. Update the model configuration in `config.py`
3. Register the model in `ModelManager`
4. Add UI controls in the sidebar

### Customizing the UI

- Modify `utils/ui_components.py` for reusable UI components
- Update `assets/styles.py` for custom CSS styling
- Edit the main application layout in `app.py`

## Features

### Core Functionality
- **Hotspot Detection**: AI-powered classification of lithography patterns
- **Domain Adaptation**: CycleGAN-based translation from synthetic to SEM images
- **Explainable AI**: Grad-CAM heatmap visualization
- **Batch Processing**: Process multiple images simultaneously

### User Interface
- Professional design with dark/light theme toggle
- Interactive sidebar with file upload and parameter controls
- Real-time visualization of results
- Download options for processed images and reports

### Model Support
- ResNet18: Convolutional neural network
- Vision Transformer: Attention-based architecture  
- EfficientNet: Efficient convolutional network
- CycleGAN: Domain adaptation network

## Performance

The application is optimized for:
- Real-time inference on CPU and GPU
- Efficient memory usage for batch processing
- Responsive UI with minimal latency
- Scalable architecture for production deployment

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Issues**: Reduce batch size in configuration
   ```python
   BATCH_SIZE = 4  # Reduce from default 32
   ```

3. **Port Already in Use**: Change the port in VS Code task or run manually
   ```bash
   streamlit run app.py --server.port=8502
   ```

4. **Model Loading Errors**: Check model file paths in configuration

### Debug Mode

Enable debug mode by setting environment variable:
```bash
export STREAMLIT_DEBUG=true
streamlit run app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For questions or issues:
- Check the troubleshooting section above
- Review the GitHub issues
- Contact the development team
