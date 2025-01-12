# Base image with Python
FROM python:3.10

# Set environment variables to avoid issues during installation
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set Hugging Face token as an environment variable (replace YOUR_HF_TOKEN with the actual token)
ENV HF_TOKEN="----"

# Log in to Hugging Face CLI using the token
RUN huggingface-cli login --token ${HF_TOKEN}

# Copy the script and other files
COPY fine_tune.py .

# Command to run the script (modify "fine_tune.py" if necessary)
CMD ["python", "fine_tune.py"]
