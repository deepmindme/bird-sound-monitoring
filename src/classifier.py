"""
Модуль классификации птиц с использованием BirdNET.

Этот модуль предоставляет удобную обёртку над библиотекой birdnetlib
для классификации видов птиц по аудиозаписям.

BirdNET — это нейронная сеть, разработанная Cornell Lab of Ornithology
и Chemnitz University of Technology для идентификации птиц по звукам.
Модель обучена на миллионах записей и распознаёт более 6000 видов птиц.

Автор: Артем Еременко
Дата: 2025
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Импорты BirdNET
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer


@dataclass
class Detection:
    """
    Результат обнаружения птицы в аудиозаписи.
    
    Attributes:
        common_name: Общепринятое название вида
        scientific_name: Научное (латинское) название
        confidence: Уверенность модели (0.0 - 1.0)
        start_time: Время начала обнаружения (секунды)
        end_time: Время окончания обнаружения (секунды)
        label: Полная метка из BirdNET
    """
    common_name: str
    scientific_name: str
    confidence: float
    start_time: float
    end_time: float
    label: str = ""
    
    def __str__(self) -> str:
        """Форматированный вывод результата обнаружения."""
        return (
            f"{self.common_name} ({self.scientific_name}) - "
            f"уверенность: {self.confidence:.1%}, "
            f"время: {self.start_time:.1f}-{self.end_time:.1f} сек"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует обнаружение в словарь."""
        return {
            'common_name': self.common_name,
            'scientific_name': self.scientific_name,
            'confidence': self.confidence,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'label': self.label
        }


@dataclass
class AnalysisResult:
    """
    Полный результат анализа аудиофайла.
    
    Attributes:
        filepath: Путь к проанализированному файлу
        detections: Список обнаруженных птиц
        duration: Длительность файла в секундах
        analysis_time: Время анализа
        location: Координаты (широта, долгота)
        date: Дата записи
    """
    filepath: str
    detections: List[Detection] = field(default_factory=list)
    duration: float = 0.0
    analysis_time: datetime = field(default_factory=datetime.now)
    location: Optional[Tuple[float, float]] = None
    date: Optional[datetime] = None
    
    @property
    def num_detections(self) -> int:
        """Количество обнаружений."""
        return len(self.detections)
    
    @property
    def unique_species(self) -> List[str]:
        """Список уникальных обнаруженных видов."""
        return list(set(d.scientific_name for d in self.detections))
    
    @property
    def top_detection(self) -> Optional[Detection]:
        """Обнаружение с наивысшей уверенностью."""
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.confidence)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Конвертирует результаты в pandas DataFrame."""
        if not self.detections:
            return pd.DataFrame()
        
        data = [d.to_dict() for d in self.detections]
        df = pd.DataFrame(data)
        df['filepath'] = self.filepath
        df['filename'] = os.path.basename(self.filepath)
        return df


class BirdClassifier:
    """
    Классификатор птиц на основе BirdNET.
    
    Этот класс предоставляет удобный интерфейс для классификации
    птиц по аудиозаписям с использованием предобученной модели BirdNET.
    
    Attributes:
        min_confidence: Минимальный порог уверенности для обнаружений
        default_lat: Широта по умолчанию (для фильтрации видов по региону)
        default_lon: Долгота по умолчанию
        
    Examples:
        >>> classifier = BirdClassifier(min_confidence=0.25)
        >>> result = classifier.analyze_single("bird_song.mp3")
        >>> for detection in result.detections:
        ...     print(detection)
    """
    
    def __init__(
        self,
        min_confidence: float = 0.25,
        default_lat: float = 55.75,  # Москва по умолчанию
        default_lon: float = 37.62
    ):
        """
        Инициализирует классификатор.
        
        Args:
            min_confidence: Минимальная уверенность для включения в результаты
                           (0.0 - 1.0, рекомендуется 0.25)
            default_lat: Широта места записи по умолчанию
            default_lon: Долгота места записи по умолчанию
        """
        self.min_confidence = min_confidence
        self.default_lat = default_lat
        self.default_lon = default_lon
        
        # Инициализируем анализатор BirdNET
        # При первом запуске модель будет загружена автоматически
        print("Загрузка модели BirdNET...")
        self.analyzer = Analyzer()
        print("Модель успешно загружена!")
    
    def analyze_single(
        self,
        filepath: str,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        date: Optional[datetime] = None,
        min_confidence: Optional[float] = None
    ) -> AnalysisResult:
        """
        Анализирует один аудиофайл.
        
        BirdNET использует координаты и дату для фильтрации видов,
        которые могут присутствовать в данном регионе в указанное время.
        Это повышает точность классификации.
        
        Args:
            filepath: Путь к аудиофайлу
            lat: Широта места записи (опционально)
            lon: Долгота места записи (опционально)
            date: Дата записи (опционально, по умолчанию сегодня)
            min_confidence: Минимальная уверенность (переопределяет значение по умолчанию)
            
        Returns:
            AnalysisResult с результатами анализа
            
        Raises:
            FileNotFoundError: Если файл не найден
            Exception: При ошибках анализа
            
        Examples:
            >>> result = classifier.analyze_single(
            ...     "recording.mp3",
            ...     lat=55.75, lon=37.62,
            ...     date=datetime(2024, 5, 15)
            ... )
        """
        # Проверяем существование файла
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        
        # Устанавливаем параметры
        use_lat = lat if lat is not None else self.default_lat
        use_lon = lon if lon is not None else self.default_lon
        use_date = date if date is not None else datetime.now()
        use_confidence = min_confidence if min_confidence is not None else self.min_confidence
        
        # Создаём объект Recording для анализа
        recording = Recording(
            self.analyzer,
            filepath,
            lat=use_lat,
            lon=use_lon,
            date=use_date,
            min_conf=use_confidence
        )
        
        # Выполняем анализ
        recording.analyze()
        
        # Преобразуем результаты в наши объекты
        detections = []
        for det in recording.detections:
            detection = Detection(
                common_name=det.get('common_name', 'Unknown'),
                scientific_name=det.get('scientific_name', 'Unknown'),
                confidence=det.get('confidence', 0.0),
                start_time=det.get('start_time', 0.0),
                end_time=det.get('end_time', 0.0),
                label=det.get('label', '')
            )
            detections.append(detection)
        
        # Формируем результат
        result = AnalysisResult(
            filepath=filepath,
            detections=detections,
            duration=recording.duration if hasattr(recording, 'duration') else 0.0,
            location=(use_lat, use_lon),
            date=use_date
        )
        
        return result
    
    def analyze_batch(
        self,
        filepaths: List[str],
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        date: Optional[datetime] = None,
        show_progress: bool = True
    ) -> List[AnalysisResult]:
        """
        Анализирует несколько аудиофайлов.
        
        Выполняет пакетную обработку с отображением прогресса.
        Полезно для анализа большого количества записей.
        
        Args:
            filepaths: Список путей к аудиофайлам
            lat: Широта места записи
            lon: Долгота места записи
            date: Дата записи
            show_progress: Показывать прогресс-бар (tqdm)
            
        Returns:
            Список AnalysisResult для каждого файла
            
        Examples:
            >>> files = ["bird1.mp3", "bird2.mp3", "bird3.mp3"]
            >>> results = classifier.analyze_batch(files)
            >>> for r in results:
            ...     print(f"{r.filepath}: {r.num_detections} обнаружений")
        """
        results = []
        
        # Создаём итератор с прогресс-баром
        iterator = tqdm(filepaths, desc="Анализ файлов", disable=not show_progress)
        
        for filepath in iterator:
            try:
                result = self.analyze_single(filepath, lat, lon, date)
                results.append(result)
                
                # Обновляем описание прогресс-бара
                if show_progress and result.num_detections > 0:
                    iterator.set_postfix({
                        'обнаружено': result.num_detections,
                        'топ': result.top_detection.common_name if result.top_detection else 'N/A'
                    })
                    
            except Exception as e:
                print(f"\nОшибка при анализе {filepath}: {e}")
                # Создаём пустой результат для файла с ошибкой
                results.append(AnalysisResult(filepath=filepath))
        
        return results


def results_to_dataframe(results: List[AnalysisResult]) -> pd.DataFrame:
    """
    Объединяет результаты анализа нескольких файлов в один DataFrame.
    
    Удобно для дальнейшего анализа и визуализации результатов.
    
    Args:
        results: Список результатов анализа
        
    Returns:
        pandas DataFrame со всеми обнаружениями
        
    Examples:
        >>> results = classifier.analyze_batch(files)
        >>> df = results_to_dataframe(results)
        >>> print(df.groupby('common_name').size().sort_values(ascending=False))
    """
    dataframes = []
    
    for result in results:
        df = result.to_dataframe()
        if not df.empty:
            dataframes.append(df)
    
    if not dataframes:
        return pd.DataFrame()
    
    # Объединяем все DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Упорядочиваем колонки
    column_order = [
        'filename', 'common_name', 'scientific_name', 
        'confidence', 'start_time', 'end_time', 'label', 'filepath'
    ]
    available_columns = [col for col in column_order if col in combined_df.columns]
    combined_df = combined_df[available_columns]
    
    return combined_df


def get_species_summary(results: List[AnalysisResult]) -> pd.DataFrame:
    """
    Создаёт сводную таблицу по обнаруженным видам.
    
    Группирует результаты по видам и вычисляет статистики:
    количество обнаружений, средняя/максимальная уверенность.
    
    Args:
        results: Список результатов анализа
        
    Returns:
        DataFrame со сводкой по видам
        
    Examples:
        >>> summary = get_species_summary(results)
        >>> print(summary.head(10))  # Топ-10 видов
    """
    df = results_to_dataframe(results)
    
    if df.empty:
        return pd.DataFrame()
    
    # Группируем по видам
    summary = df.groupby(['common_name', 'scientific_name']).agg({
        'confidence': ['count', 'mean', 'max'],
        'filename': 'nunique'
    }).round(3)
    
    # Упрощаем названия колонок
    summary.columns = [
        'num_detections',      # Количество обнаружений
        'avg_confidence',      # Средняя уверенность
        'max_confidence',      # Максимальная уверенность
        'num_files'            # В скольких файлах обнаружен
    ]
    
    # Сортируем по количеству обнаружений
    summary = summary.sort_values('num_detections', ascending=False)
    
    return summary.reset_index()


def format_detection_report(result: AnalysisResult) -> str:
    """
    Форматирует результат анализа в читаемый текстовый отчёт.
    
    Args:
        result: Результат анализа одного файла
        
    Returns:
        Форматированная строка с отчётом
        
    Examples:
        >>> result = classifier.analyze_single("bird.mp3")
        >>> print(format_detection_report(result))
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"ОТЧЁТ ОБ АНАЛИЗЕ АУДИОФАЙЛА")
    lines.append("=" * 60)
    lines.append(f"Файл: {os.path.basename(result.filepath)}")
    
    if result.location:
        lines.append(f"Координаты: {result.location[0]:.4f}, {result.location[1]:.4f}")
    
    if result.date:
        lines.append(f"Дата: {result.date.strftime('%Y-%m-%d')}")
    
    lines.append(f"Обнаружено видов: {len(result.unique_species)}")
    lines.append(f"Всего обнаружений: {result.num_detections}")
    lines.append("-" * 60)
    
    if result.detections:
        lines.append("ОБНАРУЖЕННЫЕ ВИДЫ:")
        lines.append("")
        
        # Группируем по видам
        species_detections: Dict[str, List[Detection]] = {}
        for det in result.detections:
            key = det.scientific_name
            if key not in species_detections:
                species_detections[key] = []
            species_detections[key].append(det)
        
        # Выводим информацию по каждому виду
        for species, detections in sorted(
            species_detections.items(),
            key=lambda x: max(d.confidence for d in x[1]),
            reverse=True
        ):
            best = max(detections, key=lambda d: d.confidence)
            lines.append(f"  • {best.common_name} ({species})")
            lines.append(f"    Обнаружений: {len(detections)}")
            lines.append(f"    Макс. уверенность: {best.confidence:.1%}")
            lines.append(f"    Временные метки: ", end="")
            
            times = [f"{d.start_time:.1f}-{d.end_time:.1f}с" for d in detections[:3]]
            lines[-1] = lines[-1] + ", ".join(times)
            if len(detections) > 3:
                lines[-1] += f" и ещё {len(detections) - 3}"
            lines.append("")
    else:
        lines.append("Птицы не обнаружены.")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)

