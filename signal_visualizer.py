"""
호흡 게이팅 기반 방사선 조사 시간 시각화 모듈

요구사항:
- 입력: 선량율, 레이어 전환 시간, 호흡 주기, 게이팅 진폭, DICOM RT 파일 경로
- 기능: DICOM 파일 처리, 스캔 시간 계산, 호흡/게이팅 신호 생성, 시각화

작성자: Claude
작성일: 2025.05.21
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pydicom
import logging
from typing import List, Tuple
from dataclasses import dataclass, field
import math
from datetime import datetime

log_file = f'signal_visualizer_{datetime.now().strftime("%Y%m%d")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),  # 파일 핸들러 추가
        logging.StreamHandler()  # 콘솔 출력도 유지
    ]
)
logger = logging.getLogger(__name__)

# 데이터 클래스 정의
@dataclass
class LineSegment:
    """스캔 라인 세그먼트 정보"""
    start_pos: np.ndarray
    end_pos: np.ndarray
    distance: float
    weight: float
    speed: float = 0.0
    dose_rate: float = 0.0
    mu_per_dist: float = 0.0
    raw_scan_time: float = 0.0
    rounded_scan_time: float = 0.0
    
    def __init__(self, start_pos: np.ndarray, end_pos: np.ndarray, weight: float):
        """
        라인 세그먼트 초기화
        
        Args:
            start_pos: 시작 위치 (x, y)
            end_pos: 종료 위치 (x, y)
            weight: 가중치 (MU)
        """
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.weight = weight
        self.distance = np.linalg.norm(end_pos - start_pos)

@dataclass
class Layer:
    """스캔 레이어 정보"""
    energy: float
    positions: np.ndarray
    weights: np.ndarray
    line_segments: List[LineSegment] = field(default_factory=list)
    layer_doserate: float = 0.0
    total_scan_time: float = 0.0
    
    def __init__(self, energy: float, positions: np.ndarray, weights: np.ndarray):
        """
        레이어 초기화
        
        Args:
            energy: 에너지 (MeV)
            positions: 위치 배열 [(x, y), ...]
            weights: 가중치 배열 [w1, w2, ...]
        """
        self.energy = energy
        self.positions = positions
        self.weights = weights
        self.line_segments = []
        
        # 라인 세그먼트 생성
        for i in range(1, len(positions)):
            segment = LineSegment(
                start_pos=positions[i-1],
                end_pos=positions[i],
                weight=weights[i-1]
            )
            self.line_segments.append(segment)

@dataclass
class Port:
    """포트(빔) 정보"""
    layers: List[Layer] = None
    total_scan_time: float = 0.0
    
    def __init__(self):
        """포트 초기화"""
        self.layers = []

@dataclass
class GatingPeriod:
    """게이팅 주기 정보"""
    T_on: float  # 빔 ON 구간 (초)
    T_off: float  # 빔 OFF 구간 (초)
    T_offset: float # offset value


class DicomDataProcessor:
    """DICOM 파일 처리 및 포트/레이어 정보 추출"""
    
    def __init__(self):
        """DICOM 데이터 처리기 초기화"""
        self.ports = []
    
    def load_dicom(self, file_path: str) -> bool:
        """
        DICOM 파일을 로드하고 포트/레이어 정보 추출
        
        Args:
            file_path: DICOM RT 파일 경로
            
        Returns:
            처리 성공 여부
        """
        try:
            # DICOM 파일 읽기
            d_header = pydicom.dcmread(file_path)
            
            # IonBeamSequence 확인
            if not hasattr(d_header, 'IonBeamSequence') or not d_header.IonBeamSequence:
                logger.error("DICOM 파일에 IonBeamSequence가 없습니다.")
                return False
                
            beam_sequence = list(d_header.IonBeamSequence)
            
            # 각 포트(빔) 처리
            for port_data in beam_sequence:
                port = Port()
                
                info_layer = port_data.IonControlPointSequence
                layers_info = list(info_layer)
                N_layers = len(layers_info) // 2
                
                # 각 레이어 처리
                for i_layer in range(N_layers):
                    jj = 2 * i_layer
                    
                    # 에너지 및 위치/가중치 정보
                    energy = layers_info[jj].NominalBeamEnergy
                    
                    # 라인 스캔 위치 맵
                    position_data = np.frombuffer(layers_info[jj][0x300b, 0x1094].value, dtype=np.float32)
                    positions = np.reshape(0.1*position_data, (len(position_data)//2, 2))
                    
                    # 라인 스캔 가중치
                    weights = np.frombuffer(layers_info[jj][0x300b, 0x1096].value, dtype=np.float32)
                    
                    # 레이어 객체 생성
                    layer = Layer(
                        energy=energy,
                        positions=positions,
                        weights=weights
                    )
                    
                    # 레이어 정보 저장
                    port.layers.append(layer)
                
                # 포트 정보 저장                
                self.ports.append(port)
            
            return True
                
        except Exception as e:
            logger.error(f"DICOM 파일 처리 오류: {str(e)}")
            return False
    
    def get_ports(self) -> List[Port]:
        """포트 목록 반환"""
        return self.ports


class BeamTimeCalculator:
    """빔 조사 시간 계산 및 신호 생성"""
    
    def __init__(self, time_resolution: float = 0.0001, min_doserate: float = 1.4, 
                 max_speed: float = 2000.0, min_speed: float = 10.0, 
                 doserate_table_path: str = 'LS_doserate.csv'):
        """
        빔 조사 시간 계산기 초기화
        
        Args:
            time_resolution: 시간 해상도 (초)
            min_doserate: 최소 선량율 (MU/s)
            max_speed: 최대 속도 (cm/s)
            min_speed: 최소 속도 (cm/s)
            doserate_table_path: 선량율 테이블 파일 경로
        """
        self.time_resolution = time_resolution
        self.min_doserate = min_doserate
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.doserate_table_path = doserate_table_path
        self._energy_to_doserate = {}  # 에너지별 선량율 캐시
    
    def _load_doserate_table(self) -> np.ndarray:
        """선량율 테이블을 로드"""
        try:
            return np.loadtxt(self.doserate_table_path, delimiter=',', encoding='utf-8-sig')
        except FileNotFoundError:
            logger.error("선량율 테이블 파일을 찾을 수 없습니다.")
            return np.array([])
        except ValueError:
            logger.error("선량율 테이블 형식이 올바르지 않습니다.")
            return np.array([])
        except Exception as e:
            logger.error(f"선량율 테이블 로드 오류: {e}")
            return np.array([])
    
    def get_doserate_for_energy(self, energy: float) -> float:
        """에너지에 대한 선량율 반환
        
        Args:
            energy: 에너지 값 (MeV)
            
        Returns:
            해당 에너지에 대한 선량율 (MU/s)
        """
        if energy in self._energy_to_doserate:
            return self._energy_to_doserate[energy]
            
        LS_doserate = self._load_doserate_table()
        if LS_doserate.size == 0:
            logger.warning(f"선량율 테이블이 비어있습니다. 에너지 {energy}MeV에 대해 기본값 사용.")
            return self.min_doserate
            
        # 해당 에너지에 대한 선량율 찾기
        mask = (LS_doserate[:, 0] >= energy + 0.0) & (LS_doserate[:, 0] < energy + 0.3)
        max_doserate_ind = np.where(mask)[0]
        
        if len(max_doserate_ind) > 0:
            max_doserate = LS_doserate[max_doserate_ind[0], 1]
            self._energy_to_doserate[energy] = max_doserate
            logger.debug(f"에너지 {energy}MeV에 대한 선량율: {max_doserate}MU/s")
            return max_doserate
            
        logger.warning(f"에너지 {energy}MeV에 대한 선량율을 찾을 수 없습니다. 기본값 사용.")
        return self.min_doserate

    def calculate_scan_times(self, port: Port, DR: float = None) -> List[float]:
        """
        포트의 모든 레이어에 대한 스캔 시간 계산
        
        Args:
            ports: 포트 목록
            DR: 선량율 조정 계수 (None이면 조정하지 않음)
            
        Returns:
            각 레이어별 스캔 시간 리스트
        """
        all_scan_times = []
        i_layer = 0
        for layer in port.layers:
            layer_time = self._calculate_layer_scan_time(layer, DR)
            all_scan_times.append(layer_time)
            i_layer += 1
        
        return all_scan_times
    
    def _calculate_layer_scan_time(self, layer: Layer, DR: float = None) -> float:
        """
        레이어의 스캔 시간 계산
        
        Args:
            layer: 계산할 레이어 객체
            DR: 선량율 조정 계수 (None이면 조정하지 않음)
            
        Returns:
            계산된 총 스캔 시간 (초)
        """

        machine_max_doserate = self.get_doserate_for_energy(layer.energy)
        logger.debug(f"레이어 에너지 {layer.energy}MeV, machine 최대 선량율: {machine_max_doserate}MU/s")
        
        # 시작점 제외 (0 에서 시작함.)
        segments = layer.line_segments[1:]
        num_segments = len(segments)
        
        if num_segments == 0:
            return 0.0
            
        # 배열 초기화 (벡터 연산 준비)
        distances = np.array([seg.distance for seg in segments])
        weights = np.array([seg.weight for seg in segments])

        # distances = distances[1:]
        # weights = weights[1:]
        
        # MU/cm 계산 (벡터화)
        mu_per_dist = np.zeros_like(distances, dtype=float)
        mu_per_dist = weights / distances
        
        # 임시 선량율 계산 (벡터화)
        dose_rates = self.max_speed * mu_per_dist
       
        if len(dose_rates) > 0:
            min_dr = np.min(dose_rates)
            layer.layer_doserate = max(min_dr, self.min_doserate)
        
        # DR 매개변수가 제공된 경우 추가 계산
        if DR is not None:
            speeds = distances * layer.layer_doserate / weights
            DR_eff = min(DR, max(speeds)/(1.2*min(speeds))) 

            max_dose_rate = float(np.max(dose_rates))
         
            layer_doserate_internal = min(max_dose_rate / DR_eff, machine_max_doserate)
           
            segment_times = weights / layer_doserate_internal

            speeds = distances / segment_times

            speed_mask = speeds > self.max_speed
            
            # 속도 제한 초과하는 세그먼트 무게 조정
            weights[speed_mask] = layer_doserate_internal * distances[speed_mask] / self.max_speed

            layer.layer_doserate = layer_doserate_internal
               
        crude_scan_times = weights / layer.layer_doserate

        rounded_scan_times = self.time_resolution * np.round(crude_scan_times / self.time_resolution)
        
        # 총 스캔 시간 계산
        layer.total_scan_time = sum(rounded_scan_times)
        if layer_doserate_internal != None:
            logger.info(f"레이어 {layer.energy:.2f}MeV, dr_machine: {machine_max_doserate:.2f}MU/s, dr_old: {layer.layer_doserate:.2f}MU/s, dr_max: {max_dose_rate:.2f}MU/s,  dr_corr_DR: {layer_doserate_internal:.2f}MU/s, T_scan: {layer.total_scan_time:.2f}s")
        else:
            logger.info(f"레이어 {layer.energy:.2f}MeV, dr_machine: {machine_max_doserate:.2f}MU/s, dr_old: {layer.layer_doserate:.2f}MU/s, dr_max: {max_dose_rate:.2f}MU/s,  dr_corr_DR: None, T_scan: {layer.total_scan_time:.2f}s")
       
        return layer.total_scan_time

    def generate_gating_signals(self, resp_period: float, gating_amplitude: float) -> Tuple[np.ndarray, np.ndarray, GatingPeriod]:
        """
        호흡 신호 및 게이팅 신호 생성
        
        Args:
            resp_period: 호흡 주기 (초)
            gating_amplitude: 게이팅 진폭 (0~1)
            
        Returns:
            (시간 배열, 호흡 신호, 게이팅 신호, 게이팅 주기 정보) 튜플
        """
        # 호흡 주기의 2배 시간 배열 생성 (2 주기 표시)
        time_array = np.arange(0, 2*resp_period, self.time_resolution)
        
        # 임계값 계산
        threshold = gating_amplitude
        
        # 호흡 신호 생성 (비대칭 호흡 신호 모델)
        phase_radians = 2*np.pi*time_array/resp_period - np.pi/4
        resp_signal_nn = (np.sin(phase_radians) + 
                        np.sin(phase_radians + 1.4/(2*np.pi))**2 + 
                        0.5*np.sin(phase_radians + (np.pi-5.4)/2)**2)
        resp_signal = resp_signal_nn/np.max(resp_signal_nn, 0)
        
        # 게이팅 신호 생성
        gating_signal = np.zeros_like(time_array)
        gating_indices = np.where(resp_signal < threshold)[0]
        gating_signal[gating_indices] = 1
        first_zero_index = self.time_resolution*np.argmin(gating_signal)

        # 게이팅 주기 계산
        T_on = len(gating_indices)*self.time_resolution/2  # 한 주기당
        T_off = resp_period - T_on
        
        return time_array, resp_signal, gating_signal, GatingPeriod(T_on=T_on, T_off=T_off, T_offset = first_zero_index)
    
    def simulate_beam_delivery(self, scan_times: List[float], gating_period: GatingPeriod, 
                             layer_switching_time: float) -> Tuple[float, List[Tuple[float, float, int]]]:
        """
        빔 조사 시간 시뮬레이션
        
        Args:
            scan_times: 각 레이어의 스캔 시간 리스트
            gating_period: 게이팅 주기 정보
            layer_switching_time: 레이어 전환 시간 (초)
            
        Returns:
            (총 빔 조사 시간, 레이어별 빔 조사 구간 리스트) 튜플
        """
        # 게이팅 주기 변수
        T_on = gating_period.T_on
        T_off = gating_period.T_off
        T_offset = gating_period.T_offset
        
        T_total = T_on + T_off
        logger.info(f"T_on: {T_on}, T_off: {T_off}, T_total: {T_total}")
        
        # 레이어 수
        n_layers = len(scan_times)
        
        # 결과 저장 변수
        t = 0  # 시작 시간
        beam_segments = []  # 레이어별 빔 조사 구간: (시작 시간, 종료 시간, 레이어 인덱스)
        
        # 각 레이어에 대해 시뮬레이션 수행
        for i in range(n_layers):
            # 현재 레이어의 필요 빔 조사 시간
            remaining = float(scan_times[i])
            
            # 레이어 빔 조사가 완료될 때까지 시뮬레이션
            while round(remaining, 5) > 0:
                # 현재 시간 t가 게이팅 주기 내 어느 위치인지 계산
                current_cycle_position = round((t - T_offset - T_off) % T_total, 5)
                if T_total - current_cycle_position < 1e-4:
                    current_cycle_position = 0.0
                
                # 현재 주기 내 남아있는 ON 시간 계산
                if round(T_on - current_cycle_position, 5) > 0:
                    # ON 구간에 있는 경우
                    available_time = T_on - current_cycle_position
                    
                    # 사용 가능한 시간과 남은 필요 시간 중 작은 값 사용
                    used_time = min(available_time, remaining)
                    
                    # 빔 조사 구간 저장
                    beam_segments.append((t, t + used_time, i))
                    
                    # 남은 필요 시간 감소
                    remaining -= used_time
                    
                    # 시간 증가
                    t += used_time
                else:
                    # OFF 구간에 있는 경우, 다음 ON 구간으로 점프
                    t += (T_total - current_cycle_position)
            
            # 레이어 완료 후 레이어 전환 시간 추가 (마지막 레이어가 아닌 경우에만)
            if i < n_layers - 1:
                t += layer_switching_time
        
        # logger.info(f"beam_segments: {beam_segments}")
        return t, beam_segments  # 총 치료 시간, 빔 조사 구간 리스트


class SignalVisualizer:
    """신호 시각화 클래스"""
    
    def __init__(self, fig_size=(12, 6), dpi=150):
        """
        시각화 설정 초기화
        
        Args:
            fig_size: 그래프 크기 (폭, 높이)
            dpi: 해상도
        """
        self.fig_size = fig_size
        self.dpi = dpi
        self.time_resolution = 0.0001
        
        # 그래프 설정
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 13,
            'figure.figsize': fig_size,
            'figure.dpi': dpi,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    
    # 틱 라벨 포맷팅 (1.01 -> 1, 1.02 -> 2, ...)
    def format_layer_number(self, y, pos):
        return f"{int(round((y - 1.0) * 100))}"  # 소수점 아래 2자리까지 반올림

    def setup_figure(self, time_end: float, DR: float):
        """
        그래프 초기 설정
        
        Args:
            time_end: 표시할 시간 종료 지점 (초)
            DR: 선량율 조정 계수
            
        Returns:
            생성된 figure, axis 객체
        """
        # 시간 범위를 5초 단위로 올림
        time_max = math.ceil(time_end / 5) * 5
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # 그래프 설정
        ax.set_xlim(0, time_max)
        # ax.set_xlim(0, 107)
        ax.set_ylim(0, 1.5)
        ax.set_xlabel('time (sec)')
        ax.set_ylabel('AU')
        ax.set_title(f'DR = {DR}')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))

        # 오른쪽 y축 추가
        ax2 = ax.twinx()
        ax2.set_ylim(0, 1.5)  # 필요에 따라 범위 조정
        ax2.set_ylabel('Layer number')  # 오른쪽 y축 레이블

        # 오른쪽 y축 틱 설정 (10단위: 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
        ax2.set_yticks([1.0 + i*0.1 for i in range(0, 6)])  # 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 
        ax2.yaxis.set_major_formatter(FuncFormatter(self.format_layer_number))

        # 주요 시간 간격에 점선 추가
        for t in range(0, int(time_max)+1, 10):
            ax.axvline(x=t, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        
        return fig, ax
    
    def plot_respiratory_signal(self, ax, time_array: np.ndarray, resp_signal: np.ndarray, 
                              resp_period: float, time_end: float):
        """
        호흡 신호 플롯
        
        Args:
            ax: 그래프 축 객체
            time_array: 시간 배열
            resp_signal: 호흡 신호 배열
            resp_period: 호흡 주기 (초)
            time_end: 표시할 시간 종료 지점 (초)
        """
        # 전체 시간에 맞게 호흡 신호 확장
        periods_needed = math.ceil(time_end / resp_period)
        extended_time = np.arange(0, periods_needed * resp_period, self.time_resolution)

        extended_resp = np.tile(resp_signal[:int(resp_period / self.time_resolution)], periods_needed)
        extended_resp = extended_resp[:len(extended_time)]
        
        # 호흡 신호 플롯 (파란색)
        ax.plot(extended_time, extended_resp, 'b--', label='Resp. signal', alpha=0.8, linewidth=1.0)
    
    def plot_gating_signal(self, ax, gating_signal: np.ndarray, 
                         resp_period: float, time_end: float, gating_amplitude: float):
        """
        게이팅 신호 플롯
        
        Args:
            ax: 그래프 축 객체
            time_array: 시간 배열
            gating_signal: 게이팅 신호 배열
            resp_period: 호흡 주기 (초)
            time_end: 표시할 시간 종료 지점 (초)
            gating_amplitude: 게이팅 진폭
        """

        # 전체 시간에 맞게 게이팅 신호 확장
        periods_needed = math.ceil(time_end / resp_period)
        extended_time = np.arange(0, periods_needed * resp_period, self.time_resolution)        

        extended_gating = np.tile(gating_signal[:int(resp_period / self.time_resolution)], periods_needed)
        extended_gating = extended_gating[:len(extended_time)]
        
        # 게이팅 신호 플롯 (빨간색, 0.5로 정규화)
        ax.plot(extended_time, 0.5 * extended_gating, 'r-', label='gating signal', alpha=0.8, linewidth=1.5)
        
        # 임계값 표시
        ax.axhline(y=gating_amplitude, color='purple', linestyle='--', alpha=0.5, label='gating threshold')
    
    def plot_layer_beams(self, ax, beam_segments: List[Tuple[float, float, int]], n_layers: int):
        """
        레이어별 빔 조사 구간 플롯
        
        Args:
            ax: 그래프 축 객체
            beam_segments: 빔 조사 구간 리스트 (시작 시간, 종료 시간, 레이어 인덱스)
            n_layers: 레이어 수
        """
        
        # 각 레이어별로 다른 높이와 색상으로 표시
        for start_time, end_time, layer_idx in beam_segments:
            # 높이 계산 (1.1부터 시작해 0.01씩 증가)
            height = 1.01 + layer_idx * 0.01
            
            # 레이어 빔 조사 구간 표시 (y축 방향으로 채우기)
            ax.fill_between([start_time, end_time],  # x 범위
                            [0, 0],                  # 하단 y값 (0부터)
                            [height, height],        # 상단 y값 (height까지)
                            color='green', alpha=0.5
                            )

    def add_legend(self, ax):
        """레전드 추가 및 중복 제거"""
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    def save_figure(self, fig, output_path: str, info_param: str):
        """
        그래프 저장
        
        Args:
            fig: 저장할 그래프 객체
            output_path: 저장 경로
        """
        output_file_path = os.path.join(output_path, info_param.strip() + '.jpg')

        fig.savefig(output_file_path, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"그래프가 저장되었습니다: {output_file_path}\n")

class RespiratoryGatingVisualizer:
    """호흡 게이팅 기반 방사선 조사 시간 시각화 메인 클래스"""
    def __init__(self, 
            nth_port: int,
            layer_switching_time: float,
            resp_period: float,
            gating_amplitude: float,
            dicom_file_path: str,
            DR: float = None,
            output_path: str = 'output_plots',
            doserate_table_path: str = 'LS_doserate.csv'):
        """
        호흡 게이팅 시각화 초기화
        
        Args:
            layer_switching_time: 레이어 전환 시간 (초)
            resp_period: 호흡 주기 (초)
            gating_amplitude: 게이팅 진폭 (0~1)
            dicom_file_path: DICOM RT 파일 경로
            DR: 선량율 조정 계수 (None이면 조정하지 않음)
            output_path: 결과 그래프 저장 경로
            doserate_table_path: 선량율 테이블 파일 경로
        """
        self.nth_port = nth_port
        self.layer_switching_time = layer_switching_time
        self.resp_period = resp_period
        self.gating_amplitude = gating_amplitude
        self.dicom_file_path = dicom_file_path
        self.DR = DR
        self.output_path = output_path
        self.doserate_table_path = doserate_table_path
        
        # 구성 요소 초기화
        self.dicom_processor = DicomDataProcessor()
        self.beam_calculator = BeamTimeCalculator(doserate_table_path=doserate_table_path)
        self.visualizer = SignalVisualizer()
        
        # 결과 저장 변수
        self.ports = []
        self.scan_times = []
        self.gating_period = None
        self.beam_segments = []
        self.total_time = 0.0

    def process(self) -> bool:
        """
        전체 처리 과정 실행
        
        Returns:
            처리 성공 여부
        """
        try:
            # 1. DICOM 파일 로드
            logger.info(f"DICOM 파일 로드 중: {self.dicom_file_path}")
            if not self.dicom_processor.load_dicom(self.dicom_file_path):
                logger.error("DICOM 파일 로드 실패")
                return False
            
            # 포트 정보 획득
            self.ports = self.dicom_processor.get_ports()
            if not self.ports:
                logger.error("유효한 포트 정보를 찾을 수 없습니다.")
                return False
            
            # 2. 스캔 시간 계산
            self.scan_times = self.beam_calculator.calculate_scan_times(self.ports[self.nth_port], self.DR)
            
            # 3. 호흡 신호 및 게이팅 신호 생성
            logger.info(f"호흡/게이팅 신호 생성 중 (주기: {self.resp_period}초, 진폭: {self.gating_amplitude})")
            self.time_array, self.resp_signal, self.gating_signal, self.gating_period = \
                self.beam_calculator.generate_gating_signals(self.resp_period, self.gating_amplitude)
            
            # 4. 빔 조사 시간 시뮬레이션
            logger.info(f"빔 조사 시간 시뮬레이션 중 (레이어 전환 시간: {self.layer_switching_time}초)")
            self.total_time, self.beam_segments = self.beam_calculator.simulate_beam_delivery(
                self.scan_times, self.gating_period, self.layer_switching_time
            )
            
            logger.info(f"처리 완료 - 총 치료 시간: {self.total_time:.2f}초")
            return True
            
        except Exception as e:
            logger.error(f"처리 중 오류 발생: {str(e)}")
            return False
    
    def visualize(self) -> bool:
        """
        결과 시각화 수행
        
        Returns:
            시각화 성공 여부
        """
        try:
            # 그래프 초기 설정
            fig, ax = self.visualizer.setup_figure(self.total_time, self.DR)
            
            # 호흡 신호 플롯
            self.visualizer.plot_respiratory_signal(
                ax, self.time_array, self.resp_signal, self.resp_period, self.total_time
            )
            
            # 게이팅 신호 플롯
            self.visualizer.plot_gating_signal(
                ax, self.gating_signal, self.resp_period, self.total_time, self.gating_amplitude
            )
            
            # 레이어별 빔 조사 구간 플롯
            n_layers = len(self.ports[self.nth_port].layers)
            self.visualizer.plot_layer_beams(ax, self.beam_segments, n_layers)
                    
            # 정보 텍스트 추가
            info_text = (
                f"T_LS: {self.layer_switching_time} s\n"
                f"T_R: {self.resp_period} s\n"
                f"A_G: {self.gating_amplitude}\n"
                f"T_BoT: {self.total_time:.2f} s\n"
                f"T_on: {self.gating_period.T_on:.2f} s, T_off: {self.gating_period.T_off:.2f} s"
            )
            info_param = (
                f"Port_{self.nth_port}-TLS_{self.layer_switching_time}-TR_{self.resp_period}-Amp_{self.gating_amplitude}-DR_{self.DR}"
            )
            
            # 그래프 저장
            self.visualizer.save_figure(fig, self.output_path, info_param)
            print("그래프 저장 완료")
            
            # 그래프 표시
            plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)
            
            return True
            
        except Exception as e:
            logger.error(f"시각화 중 오류 발생: {str(e)}")
            return False

def list_dicom_files(directory):
    """List all DICOM files in the specified directory."""
    dicom_files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            # Check if the file is a DICOM file
            _ = pydicom.dcmread(filepath, stop_before_pixels=True)
            dicom_files.append(filename)
        except:
            continue
    return dicom_files

# 메인 실행 코드
if __name__ == "__main__":
    for i_DR in range(10):
        DR = 20*(i_DR + 1)
        
        nth_file_of_dicom = 4

        # DICOM 파일 목록
        dicom_file_dir = "./data"
        dicom_files = list_dicom_files(dicom_file_dir)
        file_path = dicom_file_dir + "/" + dicom_files[nth_file_of_dicom]

        # 파라미터 설정
        params = {
            "nth_port": 0,
            "layer_switching_time": 2,    # 레이어 전환 시간 (초)
            "resp_period": 4.0,             # 호흡 주기 (초)
            "gating_amplitude": 0.2,        # 게이팅 진폭 (0~1)
            "dicom_file_path": file_path,   # DICOM 파일 경로
            "DR": DR,                       # 선량율 조정 계수 (선택적)
            "output_path": "output_plots",  # 결과 저장 경로
            "doserate_table_path": "LS_doserate.csv"  # 선량율 테이블 파일 경로
        }
        
        # 시각화 객체 생성 및 실행
        visualizer = RespiratoryGatingVisualizer(**params)
        
        # 처리 실행
        if visualizer.process():
            # 시각화 실행
            visualizer.visualize()
        else:
            logger.error("처리 실패, 시각화를 진행할 수 없습니다.")
