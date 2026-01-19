# Base image: Busybox (Required by Subgoal 3)
FROM busybox:latest

# Create directories
RUN mkdir -p /tmp/learningBase/train && \
    mkdir -p /tmp/learningBase/validation

# Copy data
COPY training_data.csv /tmp/learningBase/train/training_data.csv
COPY test_data.csv /tmp/learningBase/validation/test_data.csv
COPY README.md /README.md

# Keep container running
CMD ["tail", "-f", "/dev/null"]