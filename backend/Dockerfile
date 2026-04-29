# 1. Start with a lightweight Linux image running Python 3.10
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your code and your trained AI model into the container
COPY main.py .
COPY difficulty_model.joblib .

# 5. Tell Docker that this container will talk on port 8000
EXPOSE 8000

# 6. The command to start your FastAPI server when the container boots
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]