FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . /app

# Create a writable config directory for Matplotlib
RUN mkdir -p /app/.config/matplotlib

# Set environment variables for Hugging Face + Matplotlib
ENV MPLCONFIGDIR=/app/.config/matplotlib \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Run setup script (e.g., download models)
RUN chmod +x setup.sh && ./setup.sh

# Expose port for Streamlit
EXPOSE 7860

# Command to run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
