FROM busybox:latest

# Copy the model files from tmp/knowledgeBase to a safe location (not /tmp)
COPY tmp/knowledgeBase /data/knowledgeBase

# At runtime: create the directory structure and copy files to shared volume
CMD sh -c "mkdir -p /tmp/knowledgeBase && \
           cp /data/knowledgeBase/* /tmp/knowledgeBase/ && \
           echo 'Knowledge base ready at /tmp/knowledgeBase/' && \
           ls -la /tmp/knowledgeBase/ && \
           tail -f /dev/null"