# Linux image for running the Triton kernels on CPU via the interpreter (TRITON_INTERPRET=1).
# Triton has no macOS wheels, so laptop tests run here; the GPU benchmark runs on a T4.
FROM python:3.12-slim
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir triton pytest numpy matplotlib
WORKDIR /work
ENV TRITON_INTERPRET=1
CMD ["python", "-m", "pytest", "-q"]
