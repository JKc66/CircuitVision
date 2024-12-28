FROM python:3.12-slim

# Install system dependencies including OpenGL libraries
RUN apt-get update && apt-get install -y \
    ngspice \
    build-essential \
    python3-dev \
    gcc \
    libngspice0 \
    libngspice0-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables for PySpice
ENV SPICE_LIB=/usr/share/ngspice/scripts \
    PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python3.12/site-packages" \
    NGSPICE_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libngspice.so.0

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PySpice with specific configuration
RUN CFLAGS="-I/usr/include" LDFLAGS="-L/usr/lib/x86_64-linux-gnu" pip install PySpice==1.4.3 && \
    ln -s /usr/lib/x86_64-linux-gnu/libngspice.so.0 /usr/lib/libngspice.so

# Copy the rest of the application
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py"]