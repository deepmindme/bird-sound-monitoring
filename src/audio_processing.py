"""
Модуль обработки аудиофайлов для анализа звуков птиц.

Этот модуль содержит функции для загрузки, валидации и предобработки
аудиофайлов перед их анализом моделью BirdNET.

Автор: Артем Еременко
Дата: 2025
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass

import numpy as np
import librosa


# Поддерживаемые форматы аудиофайлов
SUPPORTED_FORMATS: Tuple[str, ...] = ('.mp3', '.wav', '.ogg', '.flac')

# Параметры по умолчанию для загрузки аудио
DEFAULT_SAMPLE_RATE: int = 48000  # BirdNET использует 48kHz


@dataclass
class AudioInfo:
    """
    Информация об аудиофайле.
    
    Attributes:
        filepath: Путь к файлу
        duration: Длительность в секундах
        sample_rate: Частота дискретизации
        channels: Количество каналов
        format: Формат файла
    """
    filepath: str
    duration: float
    sample_rate: int
    channels: int
    format: str
    
    def __str__(self) -> str:
        """Строковое представление информации о файле."""
        return (
            f"Файл: {os.path.basename(self.filepath)}\n"
            f"Длительность: {self.duration:.2f} сек\n"
            f"Частота дискретизации: {self.sample_rate} Hz\n"
            f"Каналов: {self.channels}\n"
            f"Формат: {self.format}"
        )


def validate_audio_file(filepath: str) -> bool:
    """
    Проверяет, является ли файл допустимым аудиофайлом.
    
    Функция проверяет:
    1. Существование файла
    2. Соответствие расширения поддерживаемым форматам
    
    Args:
        filepath: Путь к аудиофайлу
        
    Returns:
        True если файл валиден, False в противном случае
        
    Raises:
        FileNotFoundError: Если файл не существует
        ValueError: Если формат файла не поддерживается
        
    Examples:
        >>> validate_audio_file("bird_song.mp3")
        True
        >>> validate_audio_file("document.pdf")
        ValueError: Формат .pdf не поддерживается
    """
    # Проверяем существование файла
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    
    # Получаем расширение файла
    file_extension = Path(filepath).suffix.lower()
    
    # Проверяем поддерживаемые форматы
    if file_extension not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Формат {file_extension} не поддерживается. "
            f"Используйте: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    return True


def load_audio(
    filepath: str,
    target_sr: Optional[int] = None,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Загружает аудиофайл и возвращает массив numpy с данными.
    
    Эта функция использует librosa для загрузки аудио и автоматически
    конвертирует его в нужный формат для анализа.
    
    Args:
        filepath: Путь к аудиофайлу
        target_sr: Целевая частота дискретизации (по умолчанию 48kHz для BirdNET)
        mono: Если True, конвертирует в моно (рекомендуется для BirdNET)
        
    Returns:
        Кортеж из:
        - numpy массива с аудиоданными
        - частоты дискретизации
        
    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если формат не поддерживается
        
    Examples:
        >>> audio, sr = load_audio("bird_song.mp3")
        >>> print(f"Загружено {len(audio)/sr:.2f} секунд аудио")
    """
    # Валидируем файл перед загрузкой
    validate_audio_file(filepath)
    
    # Устанавливаем частоту дискретизации
    if target_sr is None:
        target_sr = DEFAULT_SAMPLE_RATE
    
    # Загружаем аудио с помощью librosa
    # librosa автоматически ресемплирует до target_sr
    audio_data, sample_rate = librosa.load(
        filepath,
        sr=target_sr,
        mono=mono
    )
    
    return audio_data, sample_rate


def get_audio_info(filepath: str) -> AudioInfo:
    """
    Получает метаданные аудиофайла без полной загрузки в память.
    
    Эта функция эффективна для получения информации о файле
    без необходимости загружать все аудиоданные.
    
    Args:
        filepath: Путь к аудиофайлу
        
    Returns:
        Объект AudioInfo с метаданными файла
        
    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если формат не поддерживается
        
    Examples:
        >>> info = get_audio_info("bird_song.mp3")
        >>> print(f"Длительность: {info.duration:.2f} сек")
    """
    # Валидируем файл
    validate_audio_file(filepath)
    
    # Получаем длительность без полной загрузки
    duration = librosa.get_duration(path=filepath)
    
    # Загружаем минимальное количество данных для получения sample_rate
    # sr=None означает использование оригинальной частоты дискретизации
    _, original_sr = librosa.load(filepath, sr=None, duration=0.1)
    
    # Определяем количество каналов
    # Для этого нужно загрузить с mono=False
    audio_stereo, _ = librosa.load(filepath, sr=None, mono=False, duration=0.1)
    channels = 1 if audio_stereo.ndim == 1 else audio_stereo.shape[0]
    
    # Получаем расширение файла
    file_format = Path(filepath).suffix.lower().replace('.', '')
    
    return AudioInfo(
        filepath=filepath,
        duration=duration,
        sample_rate=original_sr,
        channels=channels,
        format=file_format
    )


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int,
    normalize: bool = True,
    target_sr: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Предобрабатывает аудиоданные для анализа моделью BirdNET.
    
    Выполняет следующие операции:
    1. Ресемплирование до целевой частоты (если нужно)
    2. Нормализация амплитуды
    3. Удаление тишины (опционально)
    
    Args:
        audio: Numpy массив с аудиоданными
        sample_rate: Текущая частота дискретизации
        normalize: Если True, нормализует амплитуду к диапазону [-1, 1]
        target_sr: Целевая частота дискретизации
        
    Returns:
        Кортеж из предобработанного аудио и частоты дискретизации
        
    Examples:
        >>> audio, sr = load_audio("bird_song.mp3")
        >>> processed, new_sr = preprocess_audio(audio, sr, normalize=True)
    """
    processed_audio = audio.copy()
    current_sr = sample_rate
    
    # Ресемплирование если нужно
    if target_sr is not None and target_sr != sample_rate:
        processed_audio = librosa.resample(
            processed_audio,
            orig_sr=sample_rate,
            target_sr=target_sr
        )
        current_sr = target_sr
    
    # Нормализация амплитуды
    if normalize:
        # Находим максимальное абсолютное значение
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0:
            processed_audio = processed_audio / max_val
    
    return processed_audio, current_sr


def split_audio_into_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration: float = 3.0,
    overlap: float = 0.0
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Разбивает аудио на фрагменты заданной длительности.
    
    BirdNET анализирует аудио окнами по 3 секунды, поэтому эта функция
    полезна для понимания, как модель обрабатывает длинные записи.
    
    Args:
        audio: Numpy массив с аудиоданными
        sample_rate: Частота дискретизации
        chunk_duration: Длительность каждого фрагмента в секундах
        overlap: Перекрытие между фрагментами (0.0 - 1.0)
        
    Returns:
        Список кортежей: (фрагмент аудио, время начала, время конца)
        
    Examples:
        >>> audio, sr = load_audio("long_recording.mp3")
        >>> chunks = split_audio_into_chunks(audio, sr, chunk_duration=3.0)
        >>> print(f"Создано {len(chunks)} фрагментов")
    """
    # Вычисляем размер фрагмента в сэмплах
    chunk_samples = int(chunk_duration * sample_rate)
    
    # Вычисляем шаг с учётом перекрытия
    step_samples = int(chunk_samples * (1 - overlap))
    
    chunks = []
    total_samples = len(audio)
    
    # Разбиваем на фрагменты
    start_sample = 0
    while start_sample < total_samples:
        end_sample = min(start_sample + chunk_samples, total_samples)
        
        # Извлекаем фрагмент
        chunk = audio[start_sample:end_sample]
        
        # Вычисляем временные метки
        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate
        
        # Дополняем нулями если фрагмент короче требуемого
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
        
        chunks.append((chunk, start_time, end_time))
        start_sample += step_samples
    
    return chunks


def get_audio_statistics(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Вычисляет статистики аудиосигнала.
    
    Полезно для EDA (разведочного анализа данных) и понимания
    характеристик аудиозаписей.
    
    Args:
        audio: Numpy массив с аудиоданными
        sample_rate: Частота дискретизации
        
    Returns:
        Словарь со статистиками:
        - duration: длительность в секундах
        - rms_energy: среднеквадратичная энергия
        - zero_crossing_rate: частота пересечения нуля
        - spectral_centroid: спектральный центроид (яркость звука)
        
    Examples:
        >>> audio, sr = load_audio("bird_song.mp3")
        >>> stats = get_audio_statistics(audio, sr)
        >>> print(f"RMS энергия: {stats['rms_energy']:.4f}")
    """
    # Длительность
    duration = len(audio) / sample_rate
    
    # RMS энергия (громкость)
    rms = np.sqrt(np.mean(audio ** 2))
    
    # Частота пересечения нуля (характеристика шума)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    
    # Спектральный центроид (яркость)
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    )
    
    return {
        'duration': duration,
        'rms_energy': float(rms),
        'zero_crossing_rate': float(zcr),
        'spectral_centroid': float(spectral_centroid),
        'sample_rate': sample_rate,
        'num_samples': len(audio)
    }

