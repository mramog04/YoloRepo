from ultralytics import YOLO

# Ruta al archivo de configuración (editado con tus rutas)
data_yaml = '/home/marco/Escritorio/Yolo/data.yml'

# Usar modelo preentrenado de YOLOv8 nano (puedes cambiar por yolov8s.pt, yolov8m.pt, etc)
model = YOLO('yolov8n.pt')  

# Entrenar
model.train(
    data=data_yaml,
    epochs=100,       
    imgsz=640,        
    batch=16,         
    name="yoloexp_emociones"  # Nombre del experimento
)

model.export(format="onnx", weights="runs/detect/yoloexp_emociones/weights/best.pt")

best_model = YOLO("runs/detect/yoloexp_emociones/weights/best.pt")
metrics = best_model.val(data=data_yaml)
print(metrics)