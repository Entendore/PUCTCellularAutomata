import sys
import random
import math
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QSizePolicy, QSlider, QLineEdit, QPlainTextEdit, 
    QComboBox, QListWidget, QGroupBox, QTabWidget, QCheckBox, QFileDialog,
    QColorDialog
)
from PySide6.QtCore import Qt, QTimer, QRect, QThread, Signal, QPoint, QPointF
from PySide6.QtGui import QImage, QPainter, QColor, QPen, QKeyEvent, QPixmap

# ==========================================
# 1. Classic CA Patterns & Presets
# ==========================================
PATTERNS = {
    "Glider": np.array([[0,1,0],[0,0,1],[1,1,1]]),
    "R-Pentomino": np.array([[0,1,1],[1,1,0],[0,1,0]]),
    "Acorn": np.array([[0,1,0,0,0,0,0],[0,0,0,1,0,0,0],[1,1,0,0,1,1,1]]),
    "Diehard": np.array([[0,0,0,0,0,0,1,0],[1,1,0,0,0,0,0,0],[0,1,0,0,0,1,1,1]]),
    "Pulsar": np.array([
        [0,0,1,1,1,0,0,0,1,1,1,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,0,0,0,1,0,1,0,0,0,0,1], [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1], [0,0,1,1,1,0,0,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,1,1,1,0,0,0,1,1,1,0,0],
        [1,0,0,0,0,1,0,1,0,0,0,0,1], [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1], [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,0,0,0,1,1,1,0,0]
    ]),
    "Gosper Glider Gun": np.array([
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0],
        [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ])
}

RULE_PRESETS = {
    "Conway's Life": "B3/S23",
    "HighLife": "B36/S23",
    "Day & Night": "B3678/S34678",
    "Seeds": "B2/S",
    "Maze": "B3/S12345",
    "Replicator": "B1357/S1357",
    "Assimilation": "B345/S4567"
}

# ==========================================
# 2. Cellular Automata Engine
# ==========================================
class CellularAutomata:
    def __init__(self, width=100, height=100):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.trail_grid = np.zeros((height, width), dtype=np.uint8)
        self.birth_rules = {3}
        self.survival_rules = {2, 3}
        self.wrap_edges = True
        
        self.generation = 0
        self.births = 0
        self.deaths = 0
        self.history = []
        self.max_history = 200
        
        self.generations_mode = False
        self.max_age = 10

    def set_rules(self, b_set, s_set):
        self.birth_rules = b_set
        self.survival_rules = s_set

    def resize_grid(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.trail_grid = np.zeros((height, width), dtype=np.uint8)
        self.history.clear()
        self.generation = 0

    def randomize(self, density=0.3):
        self.grid = (np.random.rand(self.height, self.width) < density).astype(np.uint8) * 1
        self.trail_grid = np.zeros_like(self.grid)
        self.generation = 0
        self.history.clear()

    def random_symmetric_seed(self):
        self.clear()
        w, h = self.width // 2, self.height // 2
        seed = (np.random.rand(h, w) < 0.3).astype(np.uint8)
        top = np.concatenate((seed, np.fliplr(seed)), axis=1)
        full = np.concatenate((top, np.flipud(top)), axis=0)
        self.grid[:full.shape[0], :full.shape[1]] = full
        self.generation = 0
        self.history.clear()

    def clear(self):
        self.grid = np.zeros_like(self.grid)
        self.trail_grid = np.zeros_like(self.grid)
        self.generation = 0
        self.history.clear()

    def stamp_pattern(self, pattern, cx, cy):
        ph, pw = pattern.shape
        x0, y0 = cx - pw // 2, cy - ph // 2
        x_start, y_start = max(0, x0), max(0, y0)
        x_end, y_end = min(self.width, x0 + pw), min(self.height, y0 + ph)
        px_start, py_start = max(0, -x0), max(0, -y0)
        px_end, py_end = px_start + (x_end - x_start), py_start + (y_end - y_start)
        self.grid[y_start:y_end, x_start:x_end] = (pattern[py_start:py_end, px_start:px_end] * 1).astype(np.uint8)

    def toggle_cell(self, x, y, brush_size=1, state=True):
        for dy in range(-brush_size//2, brush_size//2 + 1):
            for dx in range(-brush_size//2, brush_size//2 + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    self.grid[ny, nx] = 1 if state else 0

    def step(self):
        binary_grid = (self.grid > 0).astype(np.uint8)
        
        if self.wrap_edges:
            neighbors = np.roll(binary_grid, 1, 0) + np.roll(binary_grid, -1, 0) + \
                         np.roll(binary_grid, 1, 1) + np.roll(binary_grid, -1, 1) + \
                         np.roll(np.roll(binary_grid, 1, 0), 1, 1) + np.roll(np.roll(binary_grid, 1, 0), -1, 1) + \
                         np.roll(np.roll(binary_grid, -1, 0), 1, 1) + np.roll(np.roll(binary_grid, -1, 0), -1, 1)
        else:
            padded = np.pad(binary_grid, 1, 'constant', constant_values=0)
            neighbors = padded[:-2, 1:-1] + padded[2:, 1:-1] + \
                         padded[1:-1, :-2] + padded[1:-1, 2:] + \
                         padded[:-2, :-2] + padded[:-2, 2:] + \
                         padded[2:, :-2] + padded[2:, 2:]
        
        new_grid = np.zeros_like(self.grid)
        birth_mask = np.isin(neighbors, list(self.birth_rules)) & (binary_grid == 0)
        
        if self.generations_mode:
            alive_mask = binary_grid == 1
            new_grid[alive_mask] = np.minimum(self.grid[alive_mask] + 1, self.max_age)
            new_grid[birth_mask] = 1
            new_grid[new_grid >= self.max_age] = 0
            self.deaths = np.sum(new_grid == 0) - np.sum(binary_grid == 0)
        else:
            surv_mask = np.isin(neighbors, list(self.survival_rules)) & (binary_grid == 1)
            self.deaths = np.sum(binary_grid & ~surv_mask)
            new_grid[surv_mask] = np.minimum(self.grid[surv_mask] + 5, 255)
            new_grid[birth_mask] = 1
        
        self.births = np.sum(birth_mask)
        
        # Update Trails (Ghost Trail Effect)
        self.trail_grid = np.maximum(0, self.trail_grid.astype(np.int16) - 15).astype(np.uint8)
        self.trail_grid[new_grid > 0] = 255
        
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        self.history.append(self.grid.copy())
        
        self.grid = new_grid
        self.generation += 1

    def rewind(self):
        if self.history:
            self.grid = self.history.pop()
            if self.generation > 0: self.generation -= 1

    def get_stats(self):
        live_cells = np.sum(self.grid > 0)
        density = live_cells / (self.width * self.height)
        activity = self.births + self.deaths
        return {"generation": self.generation, "live_cells": live_cells, "density": density, "activity": activity}

# ==========================================
# 3. PUCT Search Worker
# ==========================================
class PuCTSearchWorker(QThread):
    progress = Signal(str)
    finished = Signal(set, set, str)

    def __init__(self, iterations=40, objective="High Complexity"):
        super().__init__()
        self.iterations = iterations
        self.objective = objective
        self.rule_history = {}

    def _rules_to_string(self, b, s):
        return f"B{''.join(map(str, sorted(b)))}/S{''.join(map(str, sorted(s)))}"

    def _evaluate_rule(self, b, s):
        sim = CellularAutomata(60, 60)
        sim.set_rules(b, s)
        sim.randomize(0.4)
        
        activity_scores = []
        density_history = []
        entropy_scores = []
        
        for _ in range(30):
            sim.step()
            stats = sim.get_stats()
            activity_scores.append(stats["activity"])
            density_history.append(stats["density"])
            
            bin_grid = (sim.grid > 0).astype(np.uint8)
            if sim.wrap_edges:
                n = np.roll(bin_grid, 1, 0) + np.roll(bin_grid, -1, 0) + np.roll(bin_grid, 1, 1) + np.roll(bin_grid, -1, 1)
            else:
                n = np.pad(bin_grid, 1, 'constant', constant_values=0)
                n = n[:-2, 1:-1] + n[2:, 1:-1] + n[1:-1, :-2] + n[1:-1, 2:]
            
            counts = np.bincount(n.flatten(), minlength=9)
            probs = counts / np.sum(counts)
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            entropy_scores.append(entropy)
            
            if stats["density"] == 0.0: return 0.0
        
        final_density = density_history[-1]
        mean_activity = np.mean(activity_scores)
        mean_entropy = np.mean(entropy_scores)
        
        if self.objective == "High Complexity":
            if final_density == 1.0 or final_density == 0.0: return 0.0
            density_bonus = 1.0 - abs(final_density - 0.25)
            return (mean_entropy * 20) + (mean_activity * 0.5) + (density_bonus * 5)
        elif self.objective == "Stable/Oscillators":
            if final_density == 0.0 or final_density == 1.0: return 0.0
            activity_variance = np.var(activity_scores)
            return 100.0 - activity_variance + (final_density * 20) + (mean_entropy * 5)
        elif self.objective == "Rapid Expansion":
            growth = density_history[-1] - density_history[0]
            if final_density == 0.0: return 0.0
            return (final_density * 100) + (growth * 50) + (mean_entropy * 2)

    def run(self):
        best_score = -1
        best_b, best_s, best_str = set(), set(), "B3/S23"

        for i in range(self.iterations):
            candidates = []
            for _ in range(10):
                b_set = set(np.random.choice(range(9), size=random.randint(1, 3), replace=False))
                s_set = set(np.random.choice(range(9), size=random.randint(1, 4), replace=False))
                rule_str = self._rules_to_string(b_set, s_set)
                
                if rule_str in self.rule_history:
                    visits, total_value = self.rule_history[rule_str]
                    value = (total_value / visits) + 1.41 * np.sqrt(np.log(i+1) / (visits+1))
                else:
                    value = float('inf')
                candidates.append((value, b_set, s_set, rule_str))

            candidates.sort(key=lambda x: x[0], reverse=True)
            _, b, s, rule_str = candidates[0]
            
            score = self._evaluate_rule(b, s)
            if rule_str not in self.rule_history:
                self.rule_history[rule_str] = (0, 0.0)
            v, t = self.rule_history[rule_str]
            self.rule_history[rule_str] = (v + 1, t + score)

            if t + score > best_score:
                best_score = t + score
                best_b, best_s, best_str = b, s, rule_str

            self.progress.emit(f"Search {i+1}/{self.iterations}: {rule_str} (Score: {score:.2f})")

        self.finished.emit(best_b, best_s, best_str)

# ==========================================
# 4. Color Palettes & Visualization
# ==========================================
def build_palettes():
    palettes = {}
    p = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256): p[i] = [min(255, i*2), 0, min(255, i*4), 255]
    palettes["Cyberpunk"] = p
    
    p = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256): p[i] = [0, min(255, i*2), 0, 255]
    palettes["Matrix"] = p
    
    p = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        p[i] = [min(255, i * 3), min(255, max(0, i * 3 - 255)), min(255, max(0, i * 3 - 510)), 255]
    palettes["Fire"] = p
    
    p = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256): p[i] = [i, i, i, 255]
    palettes["Monochrome"] = p
    return palettes

class StatsGraphWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(100)
        self.setMaximumHeight(100)
        self.pop_history = []
        self.act_history = []

    def update_stats(self, pop, act):
        self.pop_history.append(pop)
        self.act_history.append(act)
        if len(self.pop_history) > 100:
            self.pop_history.pop(0)
            self.act_history.pop(0)
        self.update()

    def clear(self):
        self.pop_history.clear()
        self.act_history.clear()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = QRect(0, 0, self.width(), self.height())
        painter.fillRect(rect, QColor(20, 20, 20))
        painter.setPen(QPen(QColor(80, 80, 80), 1, Qt.DashLine))
        painter.drawLine(0, self.height()//2, self.width(), self.height()//2)

        if len(self.pop_history) < 2: return
        
        max_act = max(self.act_history) if max(self.act_history) > 0 else 1
        painter.setPen(QPen(QColor(255, 100, 100), 2))
        for i in range(1, len(self.act_history)):
            x1 = int((i-1) / 99 * self.width())
            y1 = int(self.height() - (self.act_history[i-1] / max_act) * self.height())
            x2 = int(i / 99 * self.width())
            y2 = int(self.height() - (self.act_history[i] / max_act) * self.height())
            painter.drawLine(x1, y1, x2, y2)
            
        painter.setPen(QPen(QColor(100, 100, 255), 2))
        for i in range(1, len(self.pop_history)):
            x1 = int((i-1) / 99 * self.width())
            y1 = int(self.height() - self.pop_history[i-1] * self.height())
            x2 = int(i / 99 * self.width())
            y2 = int(self.height() - self.pop_history[i] * self.height())
            painter.drawLine(x1, y1, x2, y2)


class CAWidget(QWidget):
    def __init__(self, ca_engine):
        super().__init__()
        self.ca = ca_engine
        self.setMinimumSize(500, 500)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        
        self.drawing = False
        self.erase_mode = False
        self.panning = False
        self.brush_size = 1
        self.stamp_pattern = None
        self.mouse_pos = QPoint(0, 0)
        self.show_grid_lines = False
        
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0, 0)
        
        # Visual Effects Toggles
        self.smooth_render = False
        self.bloom_effect = False
        self.show_ghost_trails = True
        
        self.palettes = build_palettes()
        self.current_palette = self.palettes["Cyberpunk"]
        self.hover_info = ("", 0, 0)

    def set_palette(self, name): 
        if name in self.palettes: self.current_palette = self.palettes[name]
    def set_custom_palette(self, c1: QColor, c2: QColor):
        p = np.zeros((256, 4), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            r = int(c1.red() * (1-t) + c2.red() * t)
            g = int(c1.green() * (1-t) + c2.green() * t)
            b = int(c1.blue() * (1-t) + c2.blue() * t)
            p[i] = [r, g, b, 255]
        self.palettes["Custom"] = p
        self.current_palette = p

    def set_stamp(self, pattern): self.stamp_pattern = pattern
    def toggle_grid_lines(self, state): self.show_grid_lines = state; self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.panning = True
            self.pan_start = event.position()
            self.pan_start_offset = QPointF(self.pan_offset)
        else:
            self.drawing = True
            self.erase_mode = (event.button() == Qt.RightButton)
            self.paint_cell(event.position().toPoint())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton: self.panning = False
        else: self.drawing = False

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        self.mouse_pos = pos
        if self.panning:
            self.pan_offset = QPointF(self.pan_start_offset.x() + (pos.x() - self.pan_start.x()),
                                      self.pan_start_offset.y() + (pos.y() - self.pan_start.y()))
        elif self.drawing:
            self.paint_cell(pos)
        
        cell_w = (self.width() / self.ca.width) * self.zoom_factor
        cell_h = (self.height() / self.ca.height) * self.zoom_factor
        gx = int((pos.x() - self.pan_offset.x()) / cell_w)
        gy = int((pos.y() - self.pan_offset.y()) / cell_h)
        
        if 0 <= gx < self.ca.width and 0 <= gy < self.ca.height:
            age = self.ca.grid[gy, gx]
            b = (self.ca.grid > 0).astype(np.uint8)
            if self.ca.wrap_edges:
                n = np.roll(b, 1, 0) + np.roll(b, -1, 0) + np.roll(b, 1, 1) + np.roll(b, -1, 1)
            else:
                n = np.pad(b, 1, 'constant', constant_values=0)
                n = n[:-2, 1:-1] + n[2:, 1:-1] + n[1:-1, :-2] + n[1:-1, 2:]
            neighbors = n[gy, gx]
            self.hover_info = (f"X:{gx} Y:{gy}", int(age), int(neighbors))
        else:
            self.hover_info = ("Out of Bounds", 0, 0)
            
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if event.modifiers() & Qt.ControlModifier:
            self.zoom_factor = max(1.0, min(10.0, self.zoom_factor + (1.0 if delta > 0 else -1.0)))
        else:
            self.brush_size = max(1, min(15, self.brush_size + (1 if delta > 0 else -1)))
        self.update()

    def paint_cell(self, pos):
        cell_w = (self.width() / self.ca.width) * self.zoom_factor
        cell_h = (self.height() / self.ca.height) * self.zoom_factor
        gx = int((pos.x() - self.pan_offset.x()) / cell_w)
        gy = int((pos.y() - self.pan_offset.y()) / cell_h)
        if self.stamp_pattern is not None:
            self.ca.stamp_pattern(self.stamp_pattern, gx, gy)
        else:
            self.ca.toggle_cell(gx, gy, self.brush_size, not self.erase_mode)

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = QRect(0, 0, self.width(), self.height())
        painter.fillRect(rect, QColor(10, 10, 10))

        # Combine Age and Trails for rendering
        # If trails are on, dead cells use trail_grid value. If off, dead is 0.
        if self.show_ghost_trails:
            render_grid = np.maximum(self.ca.grid, self.ca.trail_grid)
        else:
            render_grid = self.ca.grid
            
        flat_grid = render_grid.T.flatten()
        rgba_data = self.current_palette[flat_grid]
        
        h, w = self.ca.height, self.ca.width
        img = QImage(rgba_data.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        
        painter.setRenderHint(QPainter.Antialiasing, self.smooth_render)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, self.smooth_render)
        painter.translate(self.pan_offset)
        painter.scale(self.zoom_factor, self.zoom_factor)
        
        base_w = self.width()
        base_h = self.height()
        
        # Draw Bloom (Blurred layer)
        if self.bloom_effect:
            # To fake a bloom without heavy QGraphicsBlurEffect, we scale the image up slightly
            # and draw it with additive composition mode.
            painter.setCompositionMode(QPainter.CompositionMode_Plus)
            bloom_rect = QRect(-base_w*0.05, -base_h*0.05, base_w*1.1, base_h*1.1)
            painter.drawImage(bloom_rect, img.scaled(base_w*1.1, base_h*1.1, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        
        # Draw Main Image
        trans_mode = Qt.SmoothTransformation if self.smooth_render else Qt.FastTransformation
        painter.drawImage(QRect(0, 0, base_w, base_h), img.scaled(base_w, base_h, Qt.IgnoreAspectRatio, trans_mode))
        
        painter.resetTransform()
        
        # Draw Grid Lines
        if self.show_grid_lines and (self.ca.width <= 100 or self.zoom_factor > 2.0):
            painter.setPen(QPen(QColor(255, 255, 255, 40), 1, Qt.SolidLine))
            cell_w = (self.width() / self.ca.width) * self.zoom_factor
            cell_h = (self.height() / self.ca.height) * self.zoom_factor
            for i in range(1, self.ca.width):
                x = int(i * cell_w + self.pan_offset.x())
                painter.drawLine(x, 0, x, self.height())
            for i in range(1, self.ca.height):
                y = int(i * cell_h + self.pan_offset.y())
                painter.drawLine(0, y, self.width(), y)
        
        # Draw Brush Overlay
        if not self.drawing and self.underMouse():
            pen = QPen(QColor(255, 255, 255, 150), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            cell_w = (self.width() / self.ca.width) * self.zoom_factor
            cell_h = (self.height() / self.ca.height) * self.zoom_factor
            r = int(self.brush_size / 2 * cell_w)
            painter.drawEllipse(self.mouse_pos, r + cell_w, r + cell_h)

# ==========================================
# 5. Main GUI Window
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepMind PuCT Cellular Automata Studio - Visual Synth")
        self.resize(1200, 850)
        
        self.ca = CellularAutomata(100, 100)
        self.ca.randomize()
        
        self.auto_evolution_mode = False
        self.auto_evo_timer = QTimer(self)
        self.auto_evo_timer.timeout.connect(self.auto_evo_cycle)
        self.auto_evo_frame_count = 0
        
        self.init_ui()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_ca)
        self.timer.start(50)
        
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(200)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel (Visualization, Stats & Controls) ---
        left_panel = QVBoxLayout()
        self.ca_widget = CAWidget(self.ca)
        left_panel.addWidget(self.ca_widget)

        self.graph_widget = StatsGraphWidget()
        left_panel.addWidget(self.graph_widget)

        self.lbl_stats = QLabel("Gen: 0 | Live: 0 | Activity: 0 | Brush: 1")
        self.lbl_stats.setStyleSheet("font-weight: bold; padding: 4px;")
        left_panel.addWidget(self.lbl_stats)
        
        self.lbl_inspector = QLabel("Hover: X:0 Y:0 | Age: 0 | Neighbors: 0")
        self.lbl_inspector.setStyleSheet("color: #aaa; padding: 4px;")
        left_panel.addWidget(self.lbl_inspector)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self.update_speed)
        speed_layout.addWidget(self.speed_slider)
        left_panel.addLayout(speed_layout)

        main_layout.addLayout(left_panel, stretch=1)

        # --- Right Panel (Tabbed Controls) ---
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, stretch=0)

        # TAB 1: Simulation
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        
        gb_play = QGroupBox("Playback & Grid")
        play_layout = QHBoxLayout()
        self.btn_step_back = QPushButton("< Step")
        self.btn_step_back.clicked.connect(self.rewind)
        play_layout.addWidget(self.btn_step_back)
        
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        play_layout.addWidget(self.btn_pause)
        
        self.btn_step_fwd = QPushButton("Step >")
        self.btn_step_fwd.clicked.connect(self.step_forward)
        play_layout.addWidget(self.btn_step_fwd)
        
        self.btn_randomize = QPushButton("Random")
        self.btn_randomize.clicked.connect(self.ca.randomize)
        play_layout.addWidget(self.btn_randomize)
        
        self.btn_sym_seed = QPushButton("Sym Seed")
        self.btn_sym_seed.clicked.connect(self.ca.random_symmetric_seed)
        play_layout.addWidget(self.btn_sym_seed)
        
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.ca.clear)
        play_layout.addWidget(self.btn_clear)
        gb_play.setLayout(play_layout)
        tab1_layout.addWidget(gb_play)

        gb_grid = QGroupBox("Grid Settings")
        grid_layout = QVBoxLayout()
        grid_layout.addWidget(QLabel("Grid Resolution:"))
        self.grid_slider = QSlider(Qt.Horizontal)
        self.grid_slider.setRange(30, 300)
        self.grid_slider.setValue(100)
        self.lbl_grid_size = QLabel("100 x 100")
        self.grid_slider.valueChanged.connect(self.change_grid_size)
        grid_layout.addWidget(self.lbl_grid_size)
        grid_layout.addWidget(self.grid_slider)
        
        self.chk_wrap = QCheckBox("Wrap Edges (Toroidal)")
        self.chk_wrap.setChecked(True)
        self.chk_wrap.stateChanged.connect(lambda s: setattr(self.ca, 'wrap_edges', bool(s)))
        grid_layout.addWidget(self.chk_wrap)
        gb_grid.setLayout(grid_layout)
        tab1_layout.addWidget(gb_grid)
        
        gb_io = QGroupBox("Save / Load / Export / Import")
        io_layout = QHBoxLayout()
        self.btn_save = QPushButton("Save .npy")
        self.btn_save.clicked.connect(self.save_state)
        io_layout.addWidget(self.btn_save)
        self.btn_load = QPushButton("Load .npy")
        self.btn_load.clicked.connect(self.load_state)
        io_layout.addWidget(self.btn_load)
        self.btn_export = QPushButton("Export PNG")
        self.btn_export.clicked.connect(self.export_png)
        io_layout.addWidget(self.btn_export)
        self.btn_import_img = QPushButton("Import Image")
        self.btn_import_img.clicked.connect(self.import_image)
        io_layout.addWidget(self.btn_import_img)
        gb_io.setLayout(io_layout)
        tab1_layout.addWidget(gb_io)
        
        kb_label = QLabel("Shortcuts: Space(Pause) R(Random) C(Clear)\nLeft/Right(Step) S(Search) Ctrl+Wheel(Zoom)")
        kb_label.setStyleSheet("color: #888; font-size: 10px;")
        tab1_layout.addWidget(kb_label)
        
        tab1_layout.addStretch()
        self.tabs.addTab(tab1, "Simulation")

        # TAB 2: Visuals & Tools
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        
        gb_vis = QGroupBox("Visual FX Engine")
        vis_layout = QVBoxLayout()
        
        vis_layout.addWidget(QLabel("Color Palette:"))
        self.combo_palette = QComboBox()
        self.combo_palette.addItems(self.ca_widget.palettes.keys())
        self.combo_palette.currentTextChanged.connect(self.ca_widget.set_palette)
        vis_layout.addWidget(self.combo_palette)
        
        # Custom Color Pickers
        custom_color_layout = QHBoxLayout()
        self.btn_color1 = QPushButton("Start Color")
        self.btn_color1.setStyleSheet("background-color: #0000ff;")
        self.btn_color1.clicked.connect(lambda: self.pick_custom_color(1))
        custom_color_layout.addWidget(self.btn_color1)
        
        self.btn_color2 = QPushButton("End Color")
        self.btn_color2.setStyleSheet("background-color: #ffffff;")
        self.btn_color2.clicked.connect(lambda: self.pick_custom_color(2))
        custom_color_layout.addWidget(self.btn_color2)
        vis_layout.addLayout(custom_color_layout)

        self.chk_smooth = QCheckBox("Smooth/Liquid Rendering (Blends Pixels)")
        self.chk_smooth.setChecked(False)
        self.chk_smooth.stateChanged.connect(lambda s: setattr(self.ca_widget, 'smooth_render', bool(s)))
        vis_layout.addWidget(self.chk_smooth)
        
        self.chk_bloom = QCheckBox("Neon Glow / Bloom Effect")
        self.chk_bloom.setChecked(False)
        self.chk_bloom.stateChanged.connect(lambda s: setattr(self.ca_widget, 'bloom_effect', bool(s)))
        vis_layout.addWidget(self.chk_bloom)
        
        self.chk_trails = QCheckBox("Ghost Trails (Afterimages)")
        self.chk_trails.setChecked(True)
        self.chk_trails.stateChanged.connect(lambda s: setattr(self.ca_widget, 'show_ghost_trails', bool(s)))
        vis_layout.addWidget(self.chk_trails)
        
        self.chk_grid_lines = QCheckBox("Show Grid Lines")
        self.chk_grid_lines.setChecked(False)
        self.chk_grid_lines.stateChanged.connect(self.ca_widget.toggle_grid_lines)
        vis_layout.addWidget(self.chk_grid_lines)
        
        gb_vis.setLayout(vis_layout)
        tab2_layout.addWidget(gb_vis)
        
        gb_tools = QGroupBox("Drawing Tools")
        tools_layout = QVBoxLayout()
        tools_layout.addWidget(QLabel("Stamp Pattern (Click on Grid):"))
        self.combo_pattern = QComboBox()
        self.combo_pattern.addItems(["None"] + list(PATTERNS.keys()))
        self.combo_pattern.currentTextChanged.connect(lambda txt: self.ca_widget.set_stamp(None if txt=="None" else PATTERNS(txt)))
        tools_layout.addWidget(self.combo_pattern)
        tools_layout.addWidget(QLabel("(Mouse wheel changes brush size)"))
        gb_tools.setLayout(tools_layout)
        tab2_layout.addWidget(gb_tools)
        
        gb_gens = QGroupBox("Generations CA Mode (Lifespan)")
        gens_layout = QVBoxLayout()
        self.chk_gens = QCheckBox("Enable Generations Decay")
        self.chk_gens.setChecked(False)
        self.chk_gens.stateChanged.connect(lambda s: setattr(self.ca, 'generations_mode', bool(s)))
        gens_layout.addWidget(self.chk_gens)
        
        gens_layout.addWidget(QLabel("Max Cell Age:"))
        self.age_slider = QSlider(Qt.Horizontal)
        self.age_slider.setRange(2, 50)
        self.age_slider.setValue(10)
        self.age_slider.valueChanged.connect(lambda v: setattr(self.ca, 'max_age', v))
        gens_layout.addWidget(self.age_slider)
        gb_gens.setLayout(gens_layout)
        tab2_layout.addWidget(gb_gens)
        
        tab2_layout.addStretch()
        self.tabs.addTab(tab2, "Visuals & Tools")

        # TAB 3: Rules & AI
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        
        gb_presets = QGroupBox("Rule Presets")
        preset_layout = QVBoxLayout()
        self.combo_presets = QComboBox()
        self.combo_presets.addItems(list(RULE_PRESETS.keys()))
        self.combo_presets.currentTextChanged.connect(self.apply_preset_rule)
        preset_layout.addWidget(self.combo_presets)
        gb_presets.setLayout(preset_layout)
        tab3_layout.addWidget(gb_presets)
        
        gb_rules = QGroupBox("Manual Rules")
        rules_layout = QVBoxLayout()
        rules_layout.addWidget(QLabel("Current Rule (e.g. B3/S23):"))
        self.rule_input = QLineEdit("B3/S23")
        rules_layout.addWidget(self.rule_input)
        self.btn_apply_rule = QPushButton("Apply Manual Rule")
        self.btn_apply_rule.clicked.connect(self.apply_manual_rule)
        rules_layout.addWidget(self.btn_apply_rule)
        gb_rules.setLayout(rules_layout)
        tab3_layout.addWidget(gb_rules)

        gb_auto = QGroupBox("Autonomous AI Evolution")
        auto_layout = QVBoxLayout()
        auto_layout.addWidget(QLabel("Let the AI endlessly discover and apply rules."))
        self.btn_auto_evo = QPushButton("Start Auto-Evolution")
        self.btn_auto_evo.setStyleSheet("background-color: #e67e22; font-weight: bold;")
        self.btn_auto_evo.clicked.connect(self.toggle_auto_evolution)
        auto_layout.addWidget(self.btn_auto_evo)
        gb_auto.setLayout(auto_layout)
        tab3_layout.addWidget(gb_auto)

        gb_puct = QGroupBox("Manual PUCT Search")
        puct_layout = QVBoxLayout()
        puct_layout.addWidget(QLabel("Search Objective:"))
        self.combo_objective = QComboBox()
        self.combo_objective.addItems(["High Complexity", "Stable/Oscillators", "Rapid Expansion"])
        puct_layout.addWidget(self.combo_objective)
        
        self.btn_puct = QPushButton("Run PUCT Search Once")
        self.btn_puct.clicked.connect(lambda: self.run_puct_search(auto=False))
        puct_layout.addWidget(self.btn_puct)
        
        puct_layout.addWidget(QLabel("Search Log:"))
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(100)
        puct_layout.addWidget(self.log_output)
        
        puct_layout.addWidget(QLabel("Discovered Favorites (Double-click to apply):"))
        self.favorites_list = QListWidget()
        self.favorites_list.itemDoubleClicked.connect(self.apply_favorite_rule)
        puct_layout.addWidget(self.favorites_list)
        gb_puct.setLayout(puct_layout)
        tab3_layout.addWidget(gb_puct, stretch=1)
        self.tabs.addTab(tab3, "Rules & AI")

    def update_speed(self, val):
        self.timer.start(int(205 - (val * 2)))

    def update_ca(self):
        self.ca.step()
        self.ca_widget.update()

    def update_stats(self):
        stats = self.ca.get_stats()
        mode = "AI AUTO" if self.auto_evolution_mode else "MANUAL"
        self.lbl_stats.setText(
            f"Mode: {mode} | Gen: {stats['generation']} | "
            f"Live: {stats['live_cells']} | Activity: {stats['activity']} | Brush: {self.ca_widget.brush_size}"
        )
        self.graph_widget.update_stats(stats["density"], stats["activity"] / (self.ca.width * self.ca.height))
        
        txt, age, n = self.ca_widget.hover_info
        self.lbl_inspector.setText(f"Hover: {txt} | Age: {age} | Neighbors: {n}")

    def change_grid_size(self, val):
        self.ca.resize_grid(val, val)
        self.lbl_grid_size.setText(f"{val} x {val}")
        self.ca.randomize()
        self.graph_widget.clear()

    def toggle_pause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_pause.setText("Play")
        else:
            self.timer.start()
            self.btn_pause.setText("Pause")

    def step_forward(self):
        was_running = self.timer.isActive()
        if was_running: self.timer.stop()
        self.btn_pause.setText("Play")
        self.ca.step()
        self.ca_widget.update()

    def rewind(self):
        was_running = self.timer.isActive()
        if was_running: self.timer.stop()
        self.btn_pause.setText("Play")
        self.ca.rewind()
        self.ca_widget.update()

    def save_state(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CA State", "", "Numpy Arrays (*.npy)")
        if file_path: np.save(file_path, self.ca.grid)

    def load_state(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load CA State", "", "Numpy Arrays (*.npy)")
        if file_path:
            loaded = np.load(file_path)
            h, w = loaded.shape
            self.ca.resize_grid(w, h)
            self.ca.grid = loaded
            self.grid_slider.setValue(w)

    def export_png(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export to Image", "", "PNG Image (*.png)")
        if file_path:
            flat_grid = self.ca.grid.T.flatten()
            rgba_data = self.ca_widget.current_palette[flat_grid]
            h, w = self.ca.height, self.ca.width
            img = QImage(rgba_data.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
            img.save(file_path, "PNG")

    def import_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Image to Grid", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            # Load image, convert to grayscale, resize to fit grid, threshold
            pixmap = QPixmap(file_path)
            if pixmap.isNull(): return
            img = pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
            img = img.scaled(self.ca.width, self.ca.height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            
            # Convert QImage to numpy array
            ptr = img.bits()
            ptr.setsize(img.sizeInBytes())
            arr = np.array(ptr).reshape(self.ca.height, self.ca.width)
            
            # Threshold: pixels > 128 become 1
            self.ca.grid = (arr > 128).astype(np.uint8)
            self.ca.generation = 0
            self.ca.history.clear()

    def pick_custom_color(self, slot):
        color = QColorDialog.getColor()
        if color.isValid():
            if slot == 1:
                self.btn_color1.setStyleSheet(f"background-color: {color.name()};")
                self.custom_c1 = color
            else:
                self.btn_color2.setStyleSheet(f"background-color: {color.name()};")
                self.custom_c2 = color
            
            if hasattr(self, 'custom_c1') and hasattr(self, 'custom_c2'):
                self.ca_widget.set_custom_palette(self.custom_c1, self.custom_c2)
                self.combo_palette.setCurrentText("Custom")

    def apply_preset_rule(self, preset_name):
        rule_str = RULE_PRESETS.get(preset_name, "B3/S23")
        self.rule_input.setText(rule_str)
        self.apply_manual_rule()

    def apply_manual_rule(self):
        text = self.rule_input.text().strip().upper()
        try:
            parts = text.split('/')
            b_str = parts[0].replace('B', '')
            s_str = parts[1].replace('S', '')
            b_set = set(int(c) for c in b_str) if b_str else set()
            s_set = set(int(c) for c in s_str) if s_str else set()
            self.ca.set_rules(b_set, s_set)
            self.log_output.appendPlainText(f"Applied rule: {text}")
        except Exception:
            self.log_output.appendPlainText("Error: Invalid rule format.")

    def apply_favorite_rule(self, item):
        rule_str = item.text()
        try:
            parts = rule_str.split('/')
            b_str = parts[0].replace('B', '')
            s_str = parts[1].replace('S', '')
            b_set = set(int(c) for c in b_str) if b_str else set()
            s_set = set(int(c) for c in s_str) if s_str else set()
            self.ca.set_rules(b_set, s_set)
            self.rule_input.setText(rule_str)
            self.ca.randomize()
        except Exception:
            pass

    def toggle_auto_evolution(self):
        if self.auto_evolution_mode:
            self.auto_evolution_mode = False
            self.auto_evo_timer.stop()
            self.btn_auto_evo.setText("Start Auto-Evolution")
            self.btn_auto_evo.setStyleSheet("background-color: #e67e22; font-weight: bold;")
            self.log_output.appendPlainText("--- Auto-Evolution Stopped ---")
        else:
            self.auto_evolution_mode = True
            if not self.timer.isActive(): self.toggle_pause()
            self.auto_evo_frame_count = 0
            self.auto_evo_timer.start(1000)
            self.btn_auto_evo.setText("Stop Auto-Evolution")
            self.btn_auto_evo.setStyleSheet("background-color: #c0392b; font-weight: bold;")
            self.log_output.appendPlainText("--- Auto-Evolution Started ---")

    def auto_evo_cycle(self):
        if self.auto_evo_frame_count >= 5:
            self.auto_evo_frame_count = 0
            if not hasattr(self, 'worker') or not self.worker.isRunning():
                self.log_output.appendPlainText("AI is searching for a new rule...")
                self.run_puct_search(auto=True)
        else:
            self.auto_evo_frame_count += 1

    def run_puct_search(self, auto=False):
        self.btn_puct.setEnabled(False)
        self.log_output.appendPlainText("--- Starting PUCT Search ---")
        
        objective = self.combo_objective.currentText()
        self.worker = PuCTSearchWorker(iterations=40, objective=objective)
        self.worker.progress.connect(self.log_output.appendPlainText)
        self.worker.finished.connect(lambda b, s, r: self.on_puct_finished(b, s, r, auto))
        self.worker.start()

    def on_puct_finished(self, b, s, rule_str, auto):
        self.ca.set_rules(b, s)
        self.rule_input.setText(rule_str)
        self.log_output.appendPlainText(f"=== PUCT Selected: {rule_str} ===")
        
        items = [self.favorites_list.item(i).text() for i in range(self.favorites_list.count())]
        if rule_str not in items: self.favorites_list.addItem(rule_str)
            
        if not auto: 
            self.ca.randomize()
        else:
            action = random.choice(["random", "pattern", "morph", "morph", "sym_seed"])
            if action == "random":
                self.ca.randomize(0.15)
            elif action == "pattern":
                self.ca.clear()
                pat = random.choice(list(PATTERNS.values()))
                self.ca.stamp_pattern(pat, self.ca.width//2, self.ca.height//2)
            elif action == "sym_seed":
                self.ca.random_symmetric_seed()
            elif action == "morph":
                self.log_output.appendPlainText("--> Mutating existing structure...")
                
        self.btn_puct.setEnabled(True)

    def keyPressEvent(self, event: QKeyEvent):
        if isinstance(QApplication.focusWidget(), QLineEdit):
            super().keyPressEvent(event)
            return
            
        if event.key() == Qt.Key_Space:
            self.toggle_pause()
        elif event.key() == Qt.Key_R:
            self.ca.randomize()
        elif event.key() == Qt.Key_C:
            self.ca.clear()
        elif event.key() == Qt.Key_Right:
            self.step_forward()
        elif event.key() == Qt.Key_Left:
            self.rewind()
        elif event.key() == Qt.Key_S:
            if self.btn_puct.isEnabled(): self.run_puct_search(auto=False)
        else:
            super().keyPressEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QMainWindow { background-color: #1e1e1e; }
        QWidget { color: #d4d4d4; font-family: Consolas, monospace; }
        QTabWidget::pane { border: 1px solid #555; }
        QTabBar::tab { background: #2d2d2d; padding: 8px 12px; border: 1px solid #555; border-bottom: none; }
        QTabBar::tab:selected { background: #007acc; }
        QGroupBox { border: 1px solid #555; border-radius: 4px; margin-top: 10px; padding-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        QPushButton { background-color: #007acc; border: none; padding: 6px; border-radius: 3px; }
        QPushButton:hover { background-color: #0098ff; }
        QPushButton:disabled { background-color: #555; }
        QLineEdit, QComboBox, QListWidget, QPlainTextEdit { 
            background-color: #2d2d2d; border: 1px solid #555; padding: 4px; border-radius: 2px; 
        }
        QListWidget::item:hover { background-color: #007acc; }
        QCheckBox { padding: 4px; }
    """)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())