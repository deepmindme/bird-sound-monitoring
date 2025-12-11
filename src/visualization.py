"""
Модуль визуализации для анализа звуков птиц.

Содержит функции для создания спектрограмм, графиков уверенности
и других визуализаций результатов классификации.

Автор: Артем Еременко
Дата: 2025
"""

import os
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import librosa
import librosa.display

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Импортируем наши классы для type hints
from .classifier import AnalysisResult, Detection


def plot_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    title: str = "Спектрограмма аудиозаписи",
    figsize: Tuple[int, int] = (14, 5),
    cmap: str = "magma",
    show_colorbar: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Строит спектрограмму аудиозаписи.
    
    Спектрограмма показывает изменение частотного состава
    звука во времени. Это основной инструмент визуализации
    в биоакустике.
    
    Args:
        audio: Numpy массив с аудиоданными
        sample_rate: Частота дискретизации
        title: Заголовок графика
        figsize: Размер фигуры (ширина, высота)
        cmap: Цветовая карта matplotlib
        show_colorbar: Показывать шкалу цветов
        save_path: Путь для сохранения (если None - не сохранять)
        
    Returns:
        Объект matplotlib Figure
        
    Examples:
        >>> audio, sr = load_audio("bird_song.mp3")
        >>> fig = plot_spectrogram(audio, sr, title="Пение синицы")
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Вычисляем мел-спектрограмму
    # Мел-шкала ближе к человеческому восприятию звука
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sample_rate,
        n_mels=128,
        fmax=sample_rate // 2
    )
    
    # Конвертируем в децибелы для лучшей визуализации
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Отображаем спектрограмму
    img = librosa.display.specshow(
        mel_spec_db,
        sr=sample_rate,
        x_axis='time',
        y_axis='mel',
        ax=ax,
        cmap=cmap
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Время (секунды)', fontsize=12)
    ax.set_ylabel('Частота (Гц, мел-шкала)', fontsize=12)
    
    if show_colorbar:
        cbar = fig.colorbar(img, ax=ax, format='%+2.0f дБ')
        cbar.set_label('Амплитуда (дБ)', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Спектрограмма сохранена: {save_path}")
    
    return fig


def plot_spectrogram_with_detections(
    audio: np.ndarray,
    sample_rate: int,
    detections: List[Detection],
    title: str = "Спектрограмма с обнаруженными птицами",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Строит спектрограмму с отмеченными обнаружениями птиц.
    
    На спектрограмме выделяются временные интервалы,
    где модель обнаружила птиц.
    
    Args:
        audio: Numpy массив с аудиоданными
        sample_rate: Частота дискретизации
        detections: Список обнаружений из BirdNET
        title: Заголовок графика
        figsize: Размер фигуры
        save_path: Путь для сохранения
        
    Returns:
        Объект matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Вычисляем спектрограмму
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Отображаем спектрограмму
    img = librosa.display.specshow(
        mel_spec_db,
        sr=sample_rate,
        x_axis='time',
        y_axis='mel',
        ax=ax,
        cmap='viridis'
    )
    
    # Получаем границы осей для отрисовки прямоугольников
    duration = len(audio) / sample_rate
    y_min, y_max = ax.get_ylim()
    
    # Цвета для разных видов
    species_colors: Dict[str, str] = {}
    color_palette = plt.cm.Set1.colors
    
    # Рисуем прямоугольники для каждого обнаружения
    for i, det in enumerate(detections):
        # Назначаем цвет виду
        if det.scientific_name not in species_colors:
            color_idx = len(species_colors) % len(color_palette)
            species_colors[det.scientific_name] = color_palette[color_idx]
        
        color = species_colors[det.scientific_name]
        
        # Рисуем прямоугольник
        rect = patches.Rectangle(
            (det.start_time, y_min),
            det.end_time - det.start_time,
            y_max - y_min,
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=0.2
        )
        ax.add_patch(rect)
        
        # Добавляем подпись
        label = f"{det.common_name}\n{det.confidence:.0%}"
        ax.annotate(
            label,
            xy=(det.start_time + 0.1, y_max * 0.9),
            fontsize=8,
            color='white',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7)
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Время (секунды)', fontsize=12)
    ax.set_ylabel('Частота (Гц)', fontsize=12)
    
    fig.colorbar(img, ax=ax, format='%+2.0f дБ', label='Амплитуда (дБ)')
    
    # Добавляем легенду
    if species_colors:
        legend_patches = [
            patches.Patch(color=color, label=species, alpha=0.5)
            for species, color in species_colors.items()
        ]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confidence_timeline(
    detections: List[Detection],
    total_duration: float,
    title: str = "Уверенность обнаружений по времени",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Строит график уверенности модели по времени.
    
    Показывает, в какие моменты времени модель обнаружила
    птиц и с какой уверенностью.
    
    Args:
        detections: Список обнаружений
        total_duration: Общая длительность записи в секундах
        title: Заголовок графика
        figsize: Размер фигуры
        save_path: Путь для сохранения
        
    Returns:
        Объект matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if not detections:
        ax.text(
            0.5, 0.5, "Обнаружений нет",
            ha='center', va='center',
            fontsize=16, color='gray',
            transform=ax.transAxes
        )
        ax.set_xlim(0, total_duration)
        ax.set_ylim(0, 1)
        return fig
    
    # Группируем по видам для разных цветов
    species_data: Dict[str, List[Detection]] = {}
    for det in detections:
        if det.common_name not in species_data:
            species_data[det.common_name] = []
        species_data[det.common_name].append(det)
    
    # Создаём цветовую палитру
    colors = plt.cm.tab10.colors
    
    # Рисуем маркеры для каждого вида
    for i, (species, dets) in enumerate(species_data.items()):
        color = colors[i % len(colors)]
        
        # Средняя точка времени для каждого обнаружения
        times = [(d.start_time + d.end_time) / 2 for d in dets]
        confidences = [d.confidence for d in dets]
        
        ax.scatter(
            times, confidences,
            label=species,
            color=color,
            s=100,
            alpha=0.7,
            edgecolors='white',
            linewidths=1
        )
        
        # Рисуем горизонтальные линии для интервалов
        for det in dets:
            ax.hlines(
                det.confidence,
                det.start_time,
                det.end_time,
                colors=color,
                linewidth=3,
                alpha=0.5
            )
    
    ax.set_xlabel('Время (секунды)', fontsize=12)
    ax.set_ylabel('Уверенность модели', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, total_duration)
    ax.set_ylim(0, 1.05)
    
    # Добавляем горизонтальную линию порога
    ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Порог (25%)')
    
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_species_distribution(
    results: List[AnalysisResult],
    top_n: int = 10,
    title: str = "Распределение обнаруженных видов птиц",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Строит столбчатую диаграмму распределения видов.
    
    Показывает топ-N наиболее часто обнаруживаемых видов
    с указанием количества обнаружений.
    
    Args:
        results: Список результатов анализа
        top_n: Количество видов для отображения
        title: Заголовок графика
        figsize: Размер фигуры
        save_path: Путь для сохранения
        
    Returns:
        Объект matplotlib Figure
    """
    # Собираем все обнаружения
    all_detections = []
    for result in results:
        all_detections.extend(result.detections)
    
    if not all_detections:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "Нет данных для визуализации",
            ha='center', va='center',
            fontsize=16, color='gray',
            transform=ax.transAxes
        )
        return fig
    
    # Считаем количество обнаружений по видам
    species_counts: Dict[str, int] = {}
    for det in all_detections:
        name = det.common_name
        species_counts[name] = species_counts.get(name, 0) + 1
    
    # Сортируем и берём топ-N
    sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)
    top_species = sorted_species[:top_n]
    
    # Создаём график
    fig, ax = plt.subplots(figsize=figsize)
    
    names = [s[0] for s in top_species]
    counts = [s[1] for s in top_species]
    
    # Создаём горизонтальную столбчатую диаграмму
    bars = ax.barh(range(len(names)), counts, color=sns.color_palette("viridis", len(names)))
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()  # Топ-1 сверху
    
    ax.set_xlabel('Количество обнаружений', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Добавляем значения на столбцы
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va='center',
            fontsize=10,
            fontweight='bold'
        )
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confidence_distribution(
    results: List[AnalysisResult],
    title: str = "Распределение уверенности модели",
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Строит гистограмму распределения уверенности.
    
    Помогает понять, насколько уверенно модель
    делает предсказания на данных аудиозаписях.
    
    Args:
        results: Список результатов анализа
        title: Заголовок графика
        figsize: Размер фигуры
        save_path: Путь для сохранения
        
    Returns:
        Объект matplotlib Figure
    """
    # Собираем все значения уверенности
    confidences = []
    for result in results:
        for det in result.detections:
            confidences.append(det.confidence)
    
    if not confidences:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "Нет данных",
            ha='center', va='center',
            fontsize=16, color='gray',
            transform=ax.transAxes
        )
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Гистограмма с KDE
    ax.hist(
        confidences,
        bins=20,
        range=(0, 1),
        color='steelblue',
        alpha=0.7,
        edgecolor='white',
        label='Гистограмма'
    )
    
    # Добавляем линию среднего
    mean_conf = np.mean(confidences)
    ax.axvline(
        x=mean_conf,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Среднее: {mean_conf:.2%}'
    )
    
    ax.set_xlabel('Уверенность модели', fontsize=12)
    ax.set_ylabel('Количество обнаружений', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_summary_figure(
    audio: np.ndarray,
    sample_rate: int,
    result: AnalysisResult,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Создаёт сводную фигуру с несколькими визуализациями.
    
    Включает:
    - Спектрограмму с отметками обнаружений
    - График уверенности по времени
    - Столбчатую диаграмму видов
    
    Args:
        audio: Numpy массив с аудиоданными
        sample_rate: Частота дискретизации
        result: Результат анализа файла
        figsize: Размер фигуры
        save_path: Путь для сохранения
        
    Returns:
        Объект matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Создаём сетку подграфиков
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
    
    # 1. Спектрограмма (верхняя строка, на всю ширину)
    ax1 = fig.add_subplot(gs[0, :])
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(
        mel_spec_db,
        sr=sample_rate,
        x_axis='time',
        y_axis='mel',
        ax=ax1,
        cmap='magma'
    )
    ax1.set_title(f'Спектрограмма: {os.path.basename(result.filepath)}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Время (сек)')
    ax1.set_ylabel('Частота (Гц)')
    
    # Отмечаем обнаружения на спектрограмме
    y_min, y_max = ax1.get_ylim()
    colors = plt.cm.Set2.colors
    for i, det in enumerate(result.detections):
        color = colors[i % len(colors)]
        rect = patches.Rectangle(
            (det.start_time, y_min),
            det.end_time - det.start_time,
            y_max - y_min,
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=0.15
        )
        ax1.add_patch(rect)
    
    # 2. Временная линия уверенности (средняя строка, слева)
    ax2 = fig.add_subplot(gs[1, 0])
    if result.detections:
        duration = len(audio) / sample_rate
        for i, det in enumerate(result.detections):
            color = plt.cm.tab10.colors[i % 10]
            ax2.scatter(
                [(det.start_time + det.end_time) / 2],
                [det.confidence],
                s=80,
                color=color,
                alpha=0.7
            )
            ax2.hlines(
                det.confidence,
                det.start_time,
                det.end_time,
                colors=color,
                linewidth=3,
                alpha=0.5
            )
        ax2.set_xlim(0, duration)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel('Время (сек)')
    ax2.set_ylabel('Уверенность')
    ax2.set_title('Уверенность по времени', fontsize=11, fontweight='bold')
    ax2.axhline(y=0.25, color='red', linestyle='--', alpha=0.5)
    ax2.grid(alpha=0.3)
    
    # 3. Распределение видов (средняя строка, справа)
    ax3 = fig.add_subplot(gs[1, 1])
    if result.detections:
        species_counts: Dict[str, int] = {}
        for det in result.detections:
            species_counts[det.common_name] = species_counts.get(det.common_name, 0) + 1
        
        sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        names = [s[0][:20] for s in sorted_species]  # Ограничиваем длину
        counts = [s[1] for s in sorted_species]
        
        ax3.barh(range(len(names)), counts, color=sns.color_palette("viridis", len(names)))
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(names, fontsize=9)
        ax3.invert_yaxis()
        ax3.set_xlabel('Количество')
    ax3.set_title('Обнаруженные виды', fontsize=11, fontweight='bold')
    
    # 4. Сводная информация (нижняя строка)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Создаём текстовую сводку
    summary_text = (
        f"Файл: {os.path.basename(result.filepath)}\n"
        f"Длительность: {len(audio)/sample_rate:.1f} сек  |  "
        f"Обнаружено видов: {len(result.unique_species)}  |  "
        f"Всего обнаружений: {result.num_detections}"
    )
    
    if result.top_detection:
        summary_text += f"\n\nТоп обнаружение: {result.top_detection.common_name} " \
                       f"({result.top_detection.confidence:.0%} уверенность)"
    
    ax4.text(
        0.5, 0.5, summary_text,
        ha='center', va='center',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3)
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Сводная фигура сохранена: {save_path}")
    
    return fig


def plot_waveform(
    audio: np.ndarray,
    sample_rate: int,
    title: str = "Форма волны аудиосигнала",
    figsize: Tuple[int, int] = (14, 3),
    color: str = "steelblue",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Строит график формы волны аудиосигнала.
    
    Показывает амплитуду сигнала во времени.
    
    Args:
        audio: Numpy массив с аудиоданными
        sample_rate: Частота дискретизации
        title: Заголовок графика
        figsize: Размер фигуры
        color: Цвет линии
        save_path: Путь для сохранения
        
    Returns:
        Объект matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Создаём временную ось
    times = np.linspace(0, len(audio) / sample_rate, num=len(audio))
    
    ax.plot(times, audio, color=color, linewidth=0.5, alpha=0.8)
    ax.fill_between(times, audio, alpha=0.3, color=color)
    
    ax.set_xlabel('Время (секунды)', fontsize=12)
    ax.set_ylabel('Амплитуда', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, times[-1])
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

