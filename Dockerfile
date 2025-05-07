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
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set environment variables for PySpice with ARM64 paths
ENV PYTHONPATH=/usr/local/lib/python3.12/site-packages \
    LD_LIBRARY_PATH=/lib/aarch64-linux-gnu \
    SPICE_LIB=/usr/share/ngspice/scripts \
    NGSPICE_LIBRARY_PATH=/lib/aarch64-linux-gnu/libngspice.so

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PySpice with ARM64-specific configuration
RUN CFLAGS="-I/usr/include" LDFLAGS="-L/lib/aarch64-linux-gnu" pip install PySpice && \
    ln -s /lib/aarch64-linux-gnu/libngspice.so /usr/lib/libngspice.so

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]