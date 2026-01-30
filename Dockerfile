FROM runpod/pytorch:2.4.0

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    diffusers \
    transformers \
    accelerate \
    scipy

# Add the handler file
ADD handler.py .

# Start the handler
CMD [ "python", "-u", "handler.py" ]
