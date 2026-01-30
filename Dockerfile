# Use a high-stability, verified RunPod base image
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Define working directory
WORKDIR /

# Install core dependencies in one clean layer
RUN pip install --upgrade pip && \
    pip install runpod diffusers transformers accelerate

# Copy your handler script into the container
# Ensure your file is named exactly handler.py in your GitHub root
COPY handler.py .

# Start the worker
CMD [ "python", "-u", "/handler.py" ]
