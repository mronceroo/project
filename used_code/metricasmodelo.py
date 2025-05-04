from ultralytics import YOLO

# Carga tu modelo fine-tuneado
model = YOLO(r'C:\Users\manue\OneDrive\Documentos\Universidad\Proyecto\código\yolotry3.pt')

# Ejecuta la validación
results = model.val(
    data=r'C:\Users\manue\OneDrive\Documentos\Universidad\Proyecto\ChessBotv2.v1i.yolov8\data.yaml',
    split='val',  # o 'test'
    save_hybrid=True,
    conf=0.5,
    iou=0.5
)

# Accede a las métricas
metrics = results.results_dict
print("\nMétricas de evaluación:")
print(f"Precisión (mAP@0.5): {metrics['metrics/precision(B)']}")
print(f"Recall (mAP@0.5): {metrics['metrics/recall(B)']}")
print(f"F1-score: {2 * (metrics['metrics/precision(B)'] * metrics['metrics/recall(B)']) / (metrics['metrics/precision(B)'] + metrics['metrics/recall(B)'])}")  # F1 = 2*(P*R)/(P+R)