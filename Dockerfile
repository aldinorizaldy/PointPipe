# Step 1: Use an official lightweight PyTorch image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install system dependencies (needed for some 3D math libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy project code and weights into the container
# Copy the 'src', 'models', and 'weights' folders
COPY ./src ./src
COPY ./models ./models
COPY ./weights ./weights

# Step 6: Expose the port FastAPI runs on
EXPOSE 8000

# Step 7: The command to run the API
# Use uvicorn to serve the FastAPI app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]