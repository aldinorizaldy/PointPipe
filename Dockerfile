# Step 1: Use a slim Python base (much faster on Mac Intel)
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Updated for compatibility with Debian Trixie/Bookworm
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy only the requirements first to leverage Docker cache
COPY requirements.txt .

# Step 5: Install dependencies
# We specify the CPU-only version of Torch to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy your project files into the container
COPY src/ ./src/
COPY models/ ./models/
COPY weights/ ./weights/

# Step 7: Expose the port FastAPI will run on
EXPOSE 8000

# Step 8: Start the API using Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]