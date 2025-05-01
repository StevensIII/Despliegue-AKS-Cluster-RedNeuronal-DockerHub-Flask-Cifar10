# Usa una imagen base de Python
FROM python:3.6

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo requirements.txt
COPY requirements.txt /app/

# Copia las carpetas app y model
COPY app /app/app/
COPY model /app/model/

# Instala las dependencias
RUN pip install -r requirements.txt

# Exponer el puerto 5000 (para Flask)
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "app/app.py"]
