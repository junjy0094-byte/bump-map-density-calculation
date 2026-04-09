#!/usr/bin/env python3
"""
Bump Map Density Analyzer
- CSV 파일(헤더 포함, nx2: x좌표, y좌표)을 입력받아 범프 맵을 시각화
- 전체 영역 및 사용자 지정 사각형 영역의 범프 밀도(면적 비율 %)를 계산
- 마우스 드래그로 사각형 영역을 그려 밀도를 분석
- 여러 사각형의 합집합(union) 병합 지원
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, RectangleSelector
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union


class BumpDensityAnalyzer:
    def __init__(self, csv_path, bump_diameter, bump_pitch=None,
                 x_min=None, x_max=None, y_min=None, y_max=None):
        self.bump_diameter = bump_diameter
        self.bump_radius = bump_diameter / 2.0
        self.bump_area = np.pi * self.bump_radius ** 2
        self.bump_pitch = bump_pitch

        # CSV 로드
        df = pd.read_csv(csv_path)
        self.x = df.iloc[:, 0].values.astype(float)
        self.y = df.iloc[:, 1].values.astype(float)
        self.num_bumps = len(self.x)

        # 바운드 설정
        self._setup_bounds(x_min, x_max, y_min, y_max)

        # 사각형 영역 리스트: [(x_min, y_min, x_max, y_max), ...]
        self.rectangles = []
        self.rect_patches = []  # matplotlib patch 객체
        self.rect_labels = []   # 라벨 텍스트 객체

        # 합집합 그룹: set of rectangle indices
        self.union_groups = []  # list of sets, e.g. [{0,1}, {2,3}]

        # UI 상태
        self._drawing = False
        self._current_selector = None
        self._info_text = None
        self._mode = 'draw'  # 'draw' or 'union'
        self._union_selection = []  # 합집합 모드에서 선택된 사각형 인덱스

        # 리사이즈 상태
        self._resize_active = False
        self._resize_rect_idx = None   # 리사이즈 중인 사각형 인덱스
        self._resize_edge = None       # 'left', 'right', 'top', 'bottom'
        self._hover_rect_idx = None    # 현재 하이라이트 중인 사각형 인덱스

    def _setup_bounds(self, x_min, x_max, y_min, y_max):
        """바운드 설정: 커스텀 또는 자동(최외곽 범프 + pitch/2)"""
        if all(v is not None for v in [x_min, x_max, y_min, y_max]):
            self.bounds = (x_min, y_min, x_max, y_max)
        else:
            margin = self.bump_pitch / 2.0 if self.bump_pitch else self.bump_diameter
            self.bounds = (
                self.x.min() - margin,
                self.y.min() - margin,
                self.x.max() + margin,
                self.y.max() + margin,
            )
        self.bound_width = self.bounds[2] - self.bounds[0]
        self.bound_height = self.bounds[3] - self.bounds[1]
        self.total_area = self.bound_width * self.bound_height

    def _bumps_in_rect(self, x_min, y_min, x_max, y_max):
        """사각형 내 범프 중심 기준 범프 개수 반환"""
        mask = self._mask_in_rect(x_min, y_min, x_max, y_max)
        return int(np.sum(mask))

    def _mask_in_rect(self, x_min, y_min, x_max, y_max):
        """사각형 내 범프 마스크"""
        return (
            (self.x >= x_min) & (self.x <= x_max) &
            (self.y >= y_min) & (self.y <= y_max)
        )

    def _density(self, bump_count, area):
        """밀도 계산: 면적 비율 (%)"""
        if area <= 0:
            return 0.0
        return (bump_count * self.bump_area) / area * 100.0

    def calc_total_density(self):
        """전체 바운드 내 밀도"""
        count = self._bumps_in_rect(*self.bounds)
        return count, self._density(count, self.total_area), self.total_area

    def calc_rect_density(self, rect):
        """특정 사각형 내 밀도"""
        x0, y0, x1, y1 = rect
        area = abs(x1 - x0) * abs(y1 - y0)
        count = self._bumps_in_rect(x0, y0, x1, y1)
        return count, self._density(count, area), area

    def calc_remaining_density(self):
        """전체 영역에서 모든 사각형(합집합 고려) 영역을 뺀 나머지 밀도"""
        if not self.rectangles:
            return self.calc_total_density()

        # 면적 계산은 shapely (겹침 고려)
        all_rects_geom = self._build_all_rects_union()
        excluded_area = all_rects_geom.area

        remaining_area = self.total_area - excluded_area
        # 범프 수는 numpy 마스크 (모든 사각형의 OR)
        total_count = self._bumps_in_rect(*self.bounds)
        in_any_rect_mask = np.zeros(self.num_bumps, dtype=bool)
        for rect in self.rectangles:
            in_any_rect_mask |= self._mask_in_rect(*rect)
        in_rects_count = int(np.sum(in_any_rect_mask))
        remaining_count = total_count - in_rects_count

        return remaining_count, self._density(remaining_count, remaining_area), remaining_area

    def _build_all_rects_union(self):
        """모든 사각형(합집합 그룹 적용)의 shapely union geometry"""
        geoms = []
        for rect in self.rectangles:
            x0, y0, x1, y1 = rect
            geoms.append(shapely_box(x0, y0, x1, y1))
        return unary_union(geoms)

    def _count_bumps_in_rects(self, rect_indices):
        """여러 사각형의 합집합에 포함된 범프 수 (numpy 마스크 기반, 중심 기준)"""
        mask = np.zeros(self.num_bumps, dtype=bool)
        for idx in rect_indices:
            mask |= self._mask_in_rect(*self.rectangles[idx])
        return int(np.sum(mask))

    def _get_effective_groups(self):
        """합집합 그룹과 독립 사각형을 분리하여 반환.
        Returns: list of (label, geometry, rect_indices)
        """
        grouped_indices = set()
        for group in self.union_groups:
            grouped_indices.update(group)

        result = []

        # 합집합 그룹
        for i, group in enumerate(self.union_groups):
            geoms = [shapely_box(*self.rectangles[idx]) for idx in group]
            union_geom = unary_union(geoms)
            indices_str = "+".join([f"R{idx+1}" for idx in sorted(group)])
            label = f"Union({indices_str})"
            result.append((label, union_geom, group))

        # 독립 사각형
        for idx, rect in enumerate(self.rectangles):
            if idx not in grouped_indices:
                geom = shapely_box(*rect)
                label = f"R{idx+1}"
                result.append((label, geom, {idx}))

        return result

    # =========================================================================
    # Visualization
    # =========================================================================

    def run(self):
        """메인 GUI 실행"""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 9))
        plt.subplots_adjust(bottom=0.22, right=0.72)

        self._draw_bumps()
        self._draw_bounds()

        # 초기 전체 밀도 표시
        self._update_info()

        # 인터랙티브 사각형 그리기
        self._setup_rectangle_selector()
        self._setup_buttons()

        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (μm)')
        self.ax.set_ylabel('Y (μm)')
        self.ax.set_title('Bump Map Density Analyzer')

        plt.show()

    def _draw_bumps(self):
        """범프 그리기"""
        for xi, yi in zip(self.x, self.y):
            circle = patches.Circle(
                (xi, yi), self.bump_radius,
                linewidth=0.5, edgecolor='navy', facecolor='cornflowerblue',
                alpha=0.7
            )
            self.ax.add_patch(circle)

        # 범프 중심 점 표시
        self.ax.scatter(self.x, self.y, s=1, c='darkblue', zorder=5)

    def _draw_bounds(self):
        """바운드 영역 그리기"""
        bx0, by0, bx1, by1 = self.bounds
        bound_rect = patches.Rectangle(
            (bx0, by0), self.bound_width, self.bound_height,
            linewidth=2, edgecolor='red', facecolor='none',
            linestyle='--', label='Bound'
        )
        self.ax.add_patch(bound_rect)
        margin = max(self.bound_width, self.bound_height) * 0.05
        self.ax.set_xlim(bx0 - margin, bx1 + margin)
        self.ax.set_ylim(by0 - margin, by1 + margin)

    def _setup_rectangle_selector(self):
        """마우스 드래그로 사각형 영역 선택"""
        self._current_selector = RectangleSelector(
            self.ax, self._on_rect_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='data',
            interactive=False,
            props=dict(facecolor='yellow', edgecolor='orange',
                       alpha=0.3, linewidth=2)
        )

    def _on_rect_select(self, eclick, erelease):
        """사각형 그리기 완료 콜백"""
        if self._mode != 'draw':
            return

        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata

        # 정렬
        rx_min, rx_max = min(x0, x1), max(x0, x1)
        ry_min, ry_max = min(y0, y1), max(y0, y1)

        # 바운드 클리핑
        bx0, by0, bx1, by1 = self.bounds
        rx_min = max(rx_min, bx0)
        ry_min = max(ry_min, by0)
        rx_max = min(rx_max, bx1)
        ry_max = min(ry_max, by1)

        if rx_max <= rx_min or ry_max <= ry_min:
            return

        rect = (rx_min, ry_min, rx_max, ry_max)
        self.rectangles.append(rect)

        idx = len(self.rectangles)
        color = plt.cm.Set1(idx % 9)

        rect_patch = patches.Rectangle(
            (rx_min, ry_min), rx_max - rx_min, ry_max - ry_min,
            linewidth=2, edgecolor=color, facecolor=color,
            alpha=0.2, label=f'R{idx}'
        )
        self.ax.add_patch(rect_patch)
        self.rect_patches.append(rect_patch)

        # 라벨 텍스트
        cx = (rx_min + rx_max) / 2
        cy = (ry_min + ry_max) / 2
        txt = self.ax.text(cx, cy, f'R{idx}', ha='center', va='center',
                           fontsize=12, fontweight='bold', color=color, zorder=10)
        self.rect_labels.append(txt)

        self._update_info()
        self.fig.canvas.draw_idle()

    def _setup_buttons(self):
        """버튼 UI 설정"""
        # Undo 버튼
        ax_undo = plt.axes([0.05, 0.05, 0.12, 0.05])
        self.btn_undo = Button(ax_undo, 'Undo (Del Last)')
        self.btn_undo.on_clicked(self._on_undo)

        # Clear All 버튼
        ax_clear = plt.axes([0.19, 0.05, 0.12, 0.05])
        self.btn_clear = Button(ax_clear, 'Clear All')
        self.btn_clear.on_clicked(self._on_clear_all)

        # Union 모드 버튼
        ax_union = plt.axes([0.33, 0.05, 0.15, 0.05])
        self.btn_union = Button(ax_union, 'Union Mode: OFF')
        self.btn_union.on_clicked(self._on_toggle_union)

        # Union Apply 버튼
        ax_apply = plt.axes([0.50, 0.05, 0.15, 0.05])
        self.btn_apply = Button(ax_apply, 'Apply Union')
        self.btn_apply.on_clicked(self._on_apply_union)

        # 마우스 이벤트 (리사이즈 + 합집합 선택)
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)

    # =========================================================================
    # Edge resize
    # =========================================================================

    def _get_edge_tolerance(self):
        """현재 줌 레벨에 맞는 변 감지 허용 거리(data 좌표 기준)"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        view_width = xlim[1] - xlim[0]
        view_height = ylim[1] - ylim[0]
        return max(view_width, view_height) * 0.012

    def _detect_edge(self, mx, my):
        """마우스 위치에서 가장 가까운 사각형 변을 감지.
        Returns: (rect_idx, edge_name) or (None, None)
        """
        tol = self._get_edge_tolerance()
        best = (None, None)
        best_dist = tol

        for idx, rect in enumerate(self.rectangles):
            x0, y0, x1, y1 = rect
            # 마우스가 사각형 주변에 있는지 대략 확인
            if mx < x0 - tol or mx > x1 + tol or my < y0 - tol or my > y1 + tol:
                continue

            # 각 변까지 거리 계산 (해당 변의 범위 안에 있을 때만)
            # left edge (x=x0)
            if y0 - tol <= my <= y1 + tol:
                d = abs(mx - x0)
                if d < best_dist:
                    best_dist = d
                    best = (idx, 'left')
            # right edge (x=x1)
            if y0 - tol <= my <= y1 + tol:
                d = abs(mx - x1)
                if d < best_dist:
                    best_dist = d
                    best = (idx, 'right')
            # bottom edge (y=y0)
            if x0 - tol <= mx <= x1 + tol:
                d = abs(my - y0)
                if d < best_dist:
                    best_dist = d
                    best = (idx, 'bottom')
            # top edge (y=y1)
            if x0 - tol <= mx <= x1 + tol:
                d = abs(my - y1)
                if d < best_dist:
                    best_dist = d
                    best = (idx, 'top')

        return best

    def _on_mouse_press(self, event):
        """마우스 클릭 - 리사이즈 시작 또는 합집합 선택"""
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        # 합집합 모드
        if self._mode == 'union':
            self._on_union_click(event)
            return

        # draw 모드: 변 근처면 리사이즈 시작
        if self._mode == 'draw' and event.button == 1 and self.rectangles:
            rect_idx, edge = self._detect_edge(event.xdata, event.ydata)
            if rect_idx is not None:
                self._resize_active = True
                self._resize_rect_idx = rect_idx
                self._resize_edge = edge
                # RectangleSelector 비활성화
                if self._current_selector:
                    self._current_selector.set_active(False)
                # 리사이즈 중인 사각형 강조
                self.rect_patches[rect_idx].set_linewidth(3)
                self.rect_patches[rect_idx].set_linestyle(':')
                self.fig.canvas.draw_idle()

    def _on_mouse_move(self, event):
        """마우스 이동 - 리사이즈 중이면 사각형 업데이트"""
        if event.inaxes != self.ax or event.xdata is None:
            return

        # 리사이즈 중이면 패치만 업데이트 (밀도 계산 없음)
        if self._resize_active:
            self._do_resize(event.xdata, event.ydata)
            return

        # draw 모드: 변 근처 하이라이트 (상태 변화 시만 redraw)
        if self._mode == 'draw' and self.rectangles:
            rect_idx, edge = self._detect_edge(event.xdata, event.ydata)
            if rect_idx != self._hover_rect_idx:
                # 이전 하이라이트 복원
                if self._hover_rect_idx is not None and self._hover_rect_idx < len(self.rect_patches):
                    self.rect_patches[self._hover_rect_idx].set_linestyle('-')
                # 새 하이라이트
                if rect_idx is not None:
                    self.rect_patches[rect_idx].set_linestyle('--')
                self._hover_rect_idx = rect_idx
                self.fig.canvas.draw_idle()

    def _on_mouse_release(self, event):
        """마우스 릴리즈 - 리사이즈 종료"""
        if not self._resize_active:
            return

        idx = self._resize_rect_idx
        self._resize_active = False
        self._resize_rect_idx = None
        self._resize_edge = None

        # 시각 복원
        self.rect_patches[idx].set_linewidth(2)
        self.rect_patches[idx].set_linestyle('-')

        # RectangleSelector 재활성화
        if self._current_selector:
            self._current_selector.set_active(True)

        self._update_info()
        self.fig.canvas.draw_idle()

    def _do_resize(self, mx, my):
        """리사이즈 실행: 마우스 위치에 따라 사각형 변 이동"""
        idx = self._resize_rect_idx
        edge = self._resize_edge
        x0, y0, x1, y1 = self.rectangles[idx]
        bx0, by0, bx1, by1 = self.bounds

        min_size = self._get_edge_tolerance() * 2  # 최소 크기

        if edge == 'left':
            new_x0 = max(bx0, min(mx, x1 - min_size))
            self.rectangles[idx] = (new_x0, y0, x1, y1)
        elif edge == 'right':
            new_x1 = min(bx1, max(mx, x0 + min_size))
            self.rectangles[idx] = (x0, y0, new_x1, y1)
        elif edge == 'bottom':
            new_y0 = max(by0, min(my, y1 - min_size))
            self.rectangles[idx] = (x0, new_y0, x1, y1)
        elif edge == 'top':
            new_y1 = min(by1, max(my, y0 + min_size))
            self.rectangles[idx] = (x0, y0, x1, new_y1)

        # 패치 업데이트
        rx0, ry0, rx1, ry1 = self.rectangles[idx]
        self.rect_patches[idx].set_xy((rx0, ry0))
        self.rect_patches[idx].set_width(rx1 - rx0)
        self.rect_patches[idx].set_height(ry1 - ry0)

        # 라벨 위치 업데이트
        self.rect_labels[idx].set_position(((rx0 + rx1) / 2, (ry0 + ry1) / 2))

        self.fig.canvas.draw_idle()

    def _on_undo(self, event):
        """마지막 사각형 삭제"""
        if not self.rectangles:
            return

        self._hover_rect_idx = None
        self.rectangles.pop()
        patch = self.rect_patches.pop()
        patch.remove()
        txt = self.rect_labels.pop()
        txt.remove()

        # 합집합 그룹에서도 제거
        removed_idx = len(self.rectangles)
        self.union_groups = [
            g - {removed_idx} for g in self.union_groups
        ]
        self.union_groups = [g for g in self.union_groups if len(g) >= 2]

        self._union_selection = [s for s in self._union_selection if s < removed_idx]

        self._update_info()
        self.fig.canvas.draw_idle()

    def _on_clear_all(self, event):
        """모든 사각형 삭제"""
        self._hover_rect_idx = None
        for p in self.rect_patches:
            p.remove()
        for t in self.rect_labels:
            t.remove()
        self.rectangles.clear()
        self.rect_patches.clear()
        self.rect_labels.clear()
        self.union_groups.clear()
        self._union_selection.clear()

        self._update_info()
        self.fig.canvas.draw_idle()

    def _on_toggle_union(self, event):
        """합집합 모드 토글"""
        if self._mode == 'draw':
            self._mode = 'union'
            self._union_selection.clear()
            self.btn_union.label.set_text('Union Mode: ON')
            if self._current_selector:
                self._current_selector.set_active(False)
        else:
            self._mode = 'draw'
            self._union_selection.clear()
            self.btn_union.label.set_text('Union Mode: OFF')
            # 선택 해제 시각 효과 복원
            self._reset_rect_highlights()
            if self._current_selector:
                self._current_selector.set_active(True)

        self.fig.canvas.draw_idle()

    def _on_union_click(self, event):
        """합집합 모드에서 사각형 선택"""
        # 클릭한 위치가 어느 사각형 안인지 확인
        for idx, rect in enumerate(self.rectangles):
            x0, y0, x1, y1 = rect
            if x0 <= event.xdata <= x1 and y0 <= event.ydata <= y1:
                if idx in self._union_selection:
                    self._union_selection.remove(idx)
                    self.rect_patches[idx].set_linewidth(2)
                    self.rect_patches[idx].set_linestyle('-')
                else:
                    self._union_selection.append(idx)
                    self.rect_patches[idx].set_linewidth(4)
                    self.rect_patches[idx].set_linestyle('--')

                self.fig.canvas.draw_idle()
                break

    def _on_apply_union(self, event):
        """선택된 사각형들을 합집합으로 묶기"""
        if len(self._union_selection) < 2:
            return

        new_group = set(self._union_selection)

        # 기존 그룹과 겹치는 것이 있으면 병합
        merged = set()
        remaining_groups = []
        for group in self.union_groups:
            if group & new_group:
                merged.update(group)
            else:
                remaining_groups.append(group)
        merged.update(new_group)
        remaining_groups.append(merged)
        self.union_groups = remaining_groups

        self._union_selection.clear()
        self._reset_rect_highlights()

        # draw 모드로 복귀
        self._mode = 'draw'
        self.btn_union.label.set_text('Union Mode: OFF')
        if self._current_selector:
            self._current_selector.set_active(True)

        self._update_info()
        self.fig.canvas.draw_idle()

    def _reset_rect_highlights(self):
        """사각형 하이라이트 초기화"""
        for p in self.rect_patches:
            p.set_linewidth(2)
            p.set_linestyle('-')

    def _update_info(self):
        """밀도 정보 텍스트 업데이트"""
        if self._info_text:
            self._info_text.remove()

        lines = []
        lines.append("=" * 42)
        lines.append("  BUMP DENSITY ANALYSIS")
        lines.append("=" * 42)

        # 전체 밀도
        total_count, total_density, total_area = self.calc_total_density()
        lines.append(f"Total bumps: {self.num_bumps}")
        lines.append(f"Bump diameter: {self.bump_diameter} μm")
        lines.append(f"Bump area (each): {self.bump_area:.2f} μm²")
        lines.append(f"Bound area: {total_area:,.1f} μm²")
        lines.append("")
        lines.append(f"[Total] Density: {total_density:.4f}%")
        lines.append(f"  Bumps: {total_count}  Area: {total_area:,.1f} μm²")

        if self.rectangles:
            lines.append("")
            lines.append("-" * 42)

            effective = self._get_effective_groups()
            for label, geom, indices in effective:
                area = geom.area
                count = self._count_bumps_in_rects(indices)
                density = self._density(count, area)
                lines.append(f"[{label}] Density: {density:.4f}%")
                lines.append(f"  Bumps: {count}  Area: {area:,.1f} μm²")

            # 나머지 밀도
            lines.append("")
            rem_count, rem_density, rem_area = self.calc_remaining_density()
            lines.append(f"[Remaining] Density: {rem_density:.4f}%")
            lines.append(f"  Bumps: {rem_count}  Area: {rem_area:,.1f} μm²")

            # 각 사각형 꼭지점 좌표
            lines.append("")
            lines.append("-" * 42)
            lines.append("  VERTICES")
            lines.append("-" * 42)
            for idx, rect in enumerate(self.rectangles):
                x0, y0, x1, y1 = rect
                lines.append(f"R{idx+1}: ({x0:.1f},{y0:.1f})"
                             f" ({x1:.1f},{y0:.1f})")
                lines.append(f"    ({x0:.1f},{y1:.1f})"
                             f" ({x1:.1f},{y1:.1f})")

        lines.append("=" * 42)

        text = "\n".join(lines)
        self._info_text = self.fig.text(
            0.73, 0.95, text,
            transform=self.fig.transFigure,
            fontsize=8, fontfamily='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
        )

        # 콘솔에도 출력
        print("\n" + text)


def main():
    parser = argparse.ArgumentParser(
        description='Bump Map Density Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python bump_density_analyzer.py data.csv --diameter 100 --pitch 200
  python bump_density_analyzer.py data.csv --diameter 100 --pitch 200 --xmin 0 --xmax 1000 --ymin 0 --ymax 1000

조작 방법:
  - 마우스 드래그: 사각형 영역 그리기
  - Undo: 마지막 사각형 삭제
  - Clear All: 모든 사각형 삭제
  - Union Mode: ON/OFF 토글 후 사각형 클릭으로 선택, Apply Union으로 합집합 적용
        """
    )
    parser.add_argument('csv_file', help='범프 좌표 CSV 파일 경로 (헤더 포함, nx2)')
    parser.add_argument('--diameter', '-d', type=float, required=True,
                        help='범프 지름 (μm)')
    parser.add_argument('--pitch', '-p', type=float, default=None,
                        help='범프 피치 (μm). 자동 바운드 계산 시 사용 (미지정 시 diameter 사용)')
    parser.add_argument('--xmin', type=float, default=None, help='커스텀 X 최소 바운드')
    parser.add_argument('--xmax', type=float, default=None, help='커스텀 X 최대 바운드')
    parser.add_argument('--ymin', type=float, default=None, help='커스텀 Y 최소 바운드')
    parser.add_argument('--ymax', type=float, default=None, help='커스텀 Y 최대 바운드')

    args = parser.parse_args()

    # 커스텀 바운드: 4개 다 있거나 없거나
    custom_bounds = [args.xmin, args.xmax, args.ymin, args.ymax]
    if any(v is not None for v in custom_bounds) and not all(v is not None for v in custom_bounds):
        parser.error('커스텀 바운드를 사용하려면 --xmin, --xmax, --ymin, --ymax 모두 지정해야 합니다.')

    analyzer = BumpDensityAnalyzer(
        csv_path=args.csv_file,
        bump_diameter=args.diameter,
        bump_pitch=args.pitch,
        x_min=args.xmin, x_max=args.xmax,
        y_min=args.ymin, y_max=args.ymax,
    )
    analyzer.run()


if __name__ == '__main__':
    main()
