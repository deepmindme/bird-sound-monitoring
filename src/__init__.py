"""
Bird Sound Monitoring — пакет для мониторинга птиц по аудиозаписям.

Этот пакет предоставляет инструменты для:
- Загрузки и обработки аудиофайлов
- Классификации птиц с помощью модели BirdNET
- Визуализации результатов (спектрограммы, графики)

Пример использования:
    >>> from src import BirdClassifier, load_audio, plot_spectrogram
    >>> classifier = BirdClassifier()
    >>> result = classifier.analyze_single("bird_song.mp3")
    >>> for detection in result.detections:
    ...     print(detection)

Автор: Артем Еременко
Дата: 2025
"""

# Версия пакета
__version__ = "1.0.0"

# Импортируем основные классы и функции для удобного доступа
from .audio_processing import (
    # Функции загрузки и обработки
    load_audio,
    get_audio_info,
    preprocess_audio,
    validate_audio_file,
    split_audio_into_chunks,
    get_audio_statistics,
    # Классы данных
    AudioInfo,
    # Константы
    SUPPORTED_FORMATS,
    DEFAULT_SAMPLE_RATE,
)

from .classifier import (
    # Основной класс классификатора
    BirdClassifier,
    # Классы данных
    Detection,
    AnalysisResult,
    # Вспомогательные функции
    results_to_dataframe,
    get_species_summary,
    format_detection_report,
)

from .visualization import (
    # Функции визуализации
    plot_spectrogram,
    plot_spectrogram_with_detections,
    plot_confidence_timeline,
    plot_species_distribution,
    plot_confidence_distribution,
    create_summary_figure,
    plot_waveform,
)

# Список экспортируемых имён
__all__ = [
    # Версия
    "__version__",
    
    # Обработка аудио
    "load_audio",
    "get_audio_info",
    "preprocess_audio",
    "validate_audio_file",
    "split_audio_into_chunks",
    "get_audio_statistics",
    "AudioInfo",
    "SUPPORTED_FORMATS",
    "DEFAULT_SAMPLE_RATE",
    
    # Классификация
    "BirdClassifier",
    "Detection",
    "AnalysisResult",
    "results_to_dataframe",
    "get_species_summary",
    "format_detection_report",
    
    # Визуализация
    "plot_spectrogram",
    "plot_spectrogram_with_detections",
    "plot_confidence_timeline",
    "plot_species_distribution",
    "plot_confidence_distribution",
    "create_summary_figure",
    "plot_waveform",
]

