FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install -r requirments.txt 

EXPOSE 8501

CMD ["streamlit" , "run" , "app.py"]