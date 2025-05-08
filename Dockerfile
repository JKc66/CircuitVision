FROM python:3.12-slim

WORKDIR /app

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    ngspice \
    build-essential \
    python3-dev \
    gcc \
    libngspice0 \
    libngspice0-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONPATH=/usr/local/lib/python3.12/site-packages \
    LD_LIBRARY_PATH=/lib/aarch64-linux-gnu \
    SPICE_LIB=/usr/share/ngspice/scripts \
    NGSPICE_LIBRARY_PATH=/lib/aarch64-linux-gnu/libngspice.so

# Copy only requirements.txt and install dependencies
# This layer will be cached if requirements.txt doesn't change
COPY requirements.txt .
RUN CFLAGS="-I/usr/include" LDFLAGS="-L/lib/aarch64-linux-gnu" pip install --no-cache-dir -r requirements.txt && \
    pip install PySpice && \
    ln -s /lib/aarch64-linux-gnu/libngspice.so /usr/lib/libngspice.so

# Copy the rest of the application code
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Define the entrypoint to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]