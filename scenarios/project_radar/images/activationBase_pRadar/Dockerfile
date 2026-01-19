FROM busybox:latest

# Create directory
RUN mkdir -p /tmp/activationBase

# Copy data
COPY activation_data.csv /tmp/activationBase/activation_data.csv
COPY README.md /README.md

CMD ["tail", "-f", "/dev/null"]