# 📖 Примеры использования

Этот документ содержит подробные примеры использования системы мониторинга птиц по аудиозаписям.

---

## Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Работа с аудиофайлами](#работа-с-аудиофайлами)
3. [Классификация птиц](#классификация-птиц)
4. [Визуализация результатов](#визуализация-результатов)
5. [Пакетная обработка](#пакетная-обработка)
6. [Работа с результатами](#работа-с-результатами)
7. [Продвинутые сценарии](#продвинутые-сценарии)

---

## Быстрый старт

### Минимальный пример

```python
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime

# Загружаем модель
analyzer = Analyzer()

# Анализируем файл
recording = Recording(
    analyzer,
    "audio/bird_song.mp3",
    lat=55.75,
    lon=37.62,
    date=datetime(2024, 5, 15),
    min_conf=0.25
)
recording.analyze()

# Выводим результаты
for det in recording.detections:
    print(f"{det['common_name']}: {det['confidence']:.1%}")
```

---

## Работа с аудиофайлами

### Загрузка аудио

```python
from src.audio_processing import load_audio, get_audio_info

# Загрузка с автоматическим ресемплированием до 48kHz
audio, sr = load_audio("path/to/file.mp3")
print(f"Длительность: {len(audio)/sr:.2f} секунд")
print(f"Частота дискретизации: {sr} Hz")

# Получение метаданных без полной загрузки
info = get_audio_info("path/to/file.mp3")
print(info)
```

**Вывод:**
```
Длительность: 15.34 секунд
Частота дискретизации: 48000 Hz

Файл: file.mp3
Длительность: 15.34 сек
Частота дискретизации: 44100 Hz
Каналов: 2
Формат: mp3
```

### Валидация файлов

```python
from src.audio_processing import validate_audio_file, SUPPORTED_FORMATS

# Проверка поддерживаемых форматов
print(f"Поддерживаемые форматы: {SUPPORTED_FORMATS}")
# ('mp3', 'wav', 'ogg', 'flac')

# Валидация файла
try:
    validate_audio_file("recording.mp3")
    print("✓ Файл валиден")
except FileNotFoundError:
    print("✗ Файл не найден")
except ValueError as e:
    print(f"✗ {e}")
```

### Предобработка аудио

```python
from src.audio_processing import preprocess_audio

# Нормализация амплитуды
processed, new_sr = preprocess_audio(
    audio,
    sample_rate=sr,
    normalize=True,
    target_sr=48000
)

print(f"Новая частота: {new_sr} Hz")
print(f"Диапазон амплитуд: [{processed.min():.3f}, {processed.max():.3f}]")
```

### Разбиение на фрагменты

```python
from src.audio_processing import split_audio_into_chunks

# BirdNET анализирует окнами по 3 секунды
chunks = split_audio_into_chunks(
    audio,
    sample_rate=sr,
    chunk_duration=3.0,
    overlap=0.5  # 50% перекрытие
)

print(f"Создано фрагментов: {len(chunks)}")
for i, (chunk, start, end) in enumerate(chunks[:3]):
    print(f"  {i+1}. {start:.1f} - {end:.1f} сек")
```

### Статистики аудио

```python
from src.audio_processing import get_audio_statistics

stats = get_audio_statistics(audio, sr)
print(f"Длительность: {stats['duration']:.2f} сек")
print(f"RMS энергия: {stats['rms_energy']:.4f}")
print(f"Zero-crossing rate: {stats['zero_crossing_rate']:.4f}")
print(f"Спектральный центроид: {stats['spectral_centroid']:.1f} Hz")
```

---

## Классификация птиц

### Использование BirdClassifier

```python
from src.classifier import BirdClassifier
from datetime import datetime

# Создаём классификатор с параметрами по умолчанию
classifier = BirdClassifier(
    min_confidence=0.25,
    default_lat=55.75,  # Москва
    default_lon=37.62
)

# Анализ одного файла
result = classifier.analyze_single(
    "recording.mp3",
    lat=55.75,
    lon=37.62,
    date=datetime(2024, 5, 15)
)

# Вывод результатов
print(f"Обнаружено птиц: {result.num_detections}")
print(f"Уникальных видов: {len(result.unique_species)}")

if result.top_detection:
    top = result.top_detection
    print(f"Топ обнаружение: {top.common_name} ({top.confidence:.1%})")
```

### Детальный анализ обнаружений

```python
for detection in result.detections:
    print(f"\n{detection.common_name}")
    print(f"  Научное название: {detection.scientific_name}")
    print(f"  Уверенность: {detection.confidence:.1%}")
    print(f"  Время: {detection.start_time:.1f} - {detection.end_time:.1f} сек")
```

### Форматированный отчёт

```python
from src.classifier import format_detection_report

report = format_detection_report(result)
print(report)
```

**Вывод:**
```
============================================================
ОТЧЁТ ОБ АНАЛИЗЕ АУДИОФАЙЛА
============================================================
Файл: recording.mp3
Координаты: 55.7500, 37.6200
Дата: 2024-05-15
Обнаружено видов: 3
Всего обнаружений: 5
------------------------------------------------------------
ОБНАРУЖЕННЫЕ ВИДЫ:

  • Большая синица (Parus major)
    Обнаружений: 3
    Макс. уверенность: 85%
    Временные метки: 0.0-3.0с, 6.0-9.0с, 12.0-15.0с

  • Зяблик (Fringilla coelebs)
    Обнаружений: 2
    Макс. уверенность: 72%
    Временные метки: 3.0-6.0с, 9.0-12.0с
============================================================
```

---

## Визуализация результатов

### Спектрограмма

```python
from src.visualization import plot_spectrogram
from src.audio_processing import load_audio

audio, sr = load_audio("recording.mp3")

fig = plot_spectrogram(
    audio,
    sample_rate=sr,
    title="Пение большой синицы",
    cmap="magma",
    save_path="results/spectrogram.png"
)
```

### Спектрограмма с обнаружениями

```python
from src.visualization import plot_spectrogram_with_detections

fig = plot_spectrogram_with_detections(
    audio,
    sample_rate=sr,
    detections=result.detections,
    title="Обнаруженные виды",
    save_path="results/detections.png"
)
```

### График уверенности по времени

```python
from src.visualization import plot_confidence_timeline

duration = len(audio) / sr

fig = plot_confidence_timeline(
    detections=result.detections,
    total_duration=duration,
    title="Уверенность модели по времени",
    save_path="results/confidence.png"
)
```

### Распределение видов

```python
from src.visualization import plot_species_distribution

fig = plot_species_distribution(
    results=[result],  # Можно передать список результатов
    top_n=10,
    title="Топ-10 обнаруженных видов"
)
```

### Распределение уверенности

```python
from src.visualization import plot_confidence_distribution

fig = plot_confidence_distribution(
    results=[result],
    title="Распределение уверенности модели"
)
```

### Сводная визуализация

```python
from src.visualization import create_summary_figure

fig = create_summary_figure(
    audio=audio,
    sample_rate=sr,
    result=result,
    save_path="results/summary.png"
)
```

### Форма волны

```python
from src.visualization import plot_waveform

fig = plot_waveform(
    audio,
    sample_rate=sr,
    title="Форма волны аудиосигнала",
    color="steelblue"
)
```

---

## Пакетная обработка

### Анализ нескольких файлов

```python
from src.classifier import BirdClassifier
from pathlib import Path

classifier = BirdClassifier(min_confidence=0.25)

# Получаем список файлов
audio_files = list(Path("data/samples").glob("*.mp3"))
filepaths = [str(f) for f in audio_files]

# Пакетный анализ с прогресс-баром
results = classifier.analyze_batch(
    filepaths,
    lat=55.75,
    lon=37.62,
    show_progress=True
)

# Сводка
total_detections = sum(r.num_detections for r in results)
print(f"\nВсего файлов: {len(results)}")
print(f"Всего обнаружений: {total_detections}")
```

### Объединение результатов

```python
from src.classifier import results_to_dataframe, get_species_summary

# Все обнаружения в одном DataFrame
df = results_to_dataframe(results)
print(df.head())

# Сводка по видам
summary = get_species_summary(results)
print(summary)
```

---

## Работа с результатами

### Сохранение в CSV

```python
# Сохранение всех обнаружений
df.to_csv("results/all_detections.csv", index=False, encoding="utf-8")

# Сохранение сводки по видам
summary.to_csv("results/species_summary.csv", encoding="utf-8")
```

### Фильтрация результатов

```python
# Только высокоуверенные обнаружения (>50%)
confident = df[df['confidence'] > 0.5]
print(f"Надёжных обнаружений: {len(confident)}")

# Топ-5 видов
top_species = df.groupby('common_name').size().nlargest(5)
print(top_species)
```

### Агрегация по файлам

```python
# Количество обнаружений по файлам
by_file = df.groupby('filename').agg({
    'common_name': 'nunique',  # Уникальных видов
    'confidence': ['count', 'mean']  # Обнаружений, средняя уверенность
})
print(by_file)
```

### Временной анализ

```python
# Распределение обнаружений по времени
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.hist(df['start_time'], bins=30, edgecolor='white')
plt.xlabel('Время (сек)')
plt.ylabel('Количество обнаружений')
plt.title('Распределение обнаружений по времени')
plt.show()
```

---

## Продвинутые сценарии

### Мониторинг папки

```python
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class AudioHandler(FileSystemEventHandler):
    def __init__(self, classifier):
        self.classifier = classifier
    
    def on_created(self, event):
        if event.src_path.endswith(('.mp3', '.wav')):
            print(f"Новый файл: {event.src_path}")
            result = self.classifier.analyze_single(event.src_path)
            print(f"  Обнаружено: {result.num_detections} птиц")

# Запуск мониторинга
classifier = BirdClassifier()
handler = AudioHandler(classifier)
observer = Observer()
observer.schedule(handler, "data/incoming", recursive=False)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
```

### Интеграция с базой данных

```python
import sqlite3
from datetime import datetime

def save_to_database(result, db_path="results/detections.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Создаём таблицу если нет
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT,
            species TEXT,
            scientific_name TEXT,
            confidence REAL,
            start_time REAL,
            end_time REAL,
            analyzed_at TEXT
        )
    ''')
    
    # Сохраняем обнаружения
    for det in result.detections:
        cursor.execute('''
            INSERT INTO detections 
            (filepath, species, scientific_name, confidence, start_time, end_time, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.filepath,
            det.common_name,
            det.scientific_name,
            det.confidence,
            det.start_time,
            det.end_time,
            datetime.now().isoformat()
        ))
    
    conn.commit()
    conn.close()

# Использование
save_to_database(result)
```

### Экспорт в JSON

```python
import json

def export_to_json(results, output_path):
    data = []
    for result in results:
        data.append({
            'filepath': result.filepath,
            'duration': result.duration,
            'location': result.location,
            'date': result.date.isoformat() if result.date else None,
            'detections': [det.to_dict() for det in result.detections]
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

export_to_json(results, "results/analysis.json")
```

### Фильтрация по региону

```python
# Анализ с координатами разных регионов
regions = {
    'moscow': (55.75, 37.62),
    'spb': (59.93, 30.31),
    'kazan': (55.79, 49.12)
}

for region_name, (lat, lon) in regions.items():
    result = classifier.analyze_single(
        "recording.mp3",
        lat=lat,
        lon=lon
    )
    print(f"{region_name}: {result.num_detections} обнаружений")
```

---

## Советы и лучшие практики

### Качество записи

1. **Минимизируйте шум** — записывайте в тихих условиях
2. **Избегайте ветра** — используйте ветрозащиту на микрофоне
3. **Записывайте утром** — максимальная активность птиц с 5:00 до 10:00
4. **Используйте направленные микрофоны** — лучшее отношение сигнал/шум

### Параметры анализа

1. **Порог уверенности:**
   - 0.25 — для разведки (больше обнаружений, больше ложных срабатываний)
   - 0.5 — для исследований (баланс)
   - 0.75 — для публикаций (высокая надёжность)

2. **Координаты:**
   - Указывайте точные координаты для лучшей фильтрации
   - Модель исключает виды, не характерные для региона

3. **Дата:**
   - Учитывает сезонность и миграции
   - Влияет на список возможных видов

### Интерпретация результатов

1. **Проверяйте спорные обнаружения** по спектрограмме
2. **Учитывайте контекст** — редкие виды требуют дополнительной проверки
3. **Сравнивайте с референсами** на Xeno-Canto

---

*Документация актуальна для версии 1.0.0*

---

**Автор:** Артем Еременко, 2025

