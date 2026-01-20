FROM busybox:latest

# Copy the CSV file to a safe location in the image (not /tmp)
COPY activation_data.csv /data/activation_data.csv

# At runtime: create the directory structure and copy file to shared volume
CMD sh -c "mkdir -p /tmp/activationBase && \
           cp /data/activation_data.csv /tmp/activationBase/activation_data.csv && \
           echo 'Activation data ready at /tmp/activationBase/activation_data.csv' && \
           ls -la /tmp/activationBase/ && \
           tail -f /dev/null"