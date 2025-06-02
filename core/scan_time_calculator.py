"""
GPU 기반 스캔 시간 계산기
"""
import cupy as cp
import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
import os

# 로컬 모듈 임포트
from config import Config
from core.gpu_utils import GPUMemoryManager
from data_models.dicom_structures import Layer, Port

logger = logging.getLogger(__name__)

class ScanTimeCalculator:
    """스캔 시간 계산 전용 클래스"""
    
    def __init__(self, 
               min_doserate: float = Config.MIN_DOSERATE, 
               max_speed: float = Config.MAX_SPEED, 
               min_speed: float = Config.MIN_SPEED,
               time_resolution: float = Config.TIME_RESOLUTION,
               doserate_table_path: str = None):
        """
        스캔 시간 계산기 초기화
        
        Args:
            min_doserate: 최소 선량율 (MU/s)
            max_speed: 최대 속도 (cm/s)
            min_speed: 최소 속도 (cm/s)
            time_resolution: 시간 해상도 (s)
            doserate_table_path: 선량율 테이블 경로 (선택적)
        """
        self.min_doserate = min_doserate
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.time_resolution = time_resolution
        self.doserate_table_path = doserate_table_path
        
        # 캐시 초기화
        self._energy_to_doserate = {}
        self.doserate_table_gpu = None  # GPU 선량율 테이블 초기화
        
        # GPU 메모리 관리자
        self.gpu_manager = GPUMemoryManager()
        
        # 선량율 테이블 로드 및 GPU로 전송
        try:
            loaded_table = self._load_doserate_table()
            if loaded_table is not None and loaded_table.size > 0:
                self.doserate_table_gpu = self.gpu_manager.array_to_gpu(loaded_table)
                if self.doserate_table_gpu is not None:
                    logger.info("선량율 테이블 GPU로 성공적으로 전송됨.")
                else:
                    logger.warning("선량율 테이블 GPU 전송 실패.")
            elif loaded_table is None: # 명시적으로 None을 반환하는 경우 (파일 못찾거나 할때)
                logger.warning("선량율 테이블 로드 실패 (테이블이 None). GPU 전송 시도 안함.")
            else: # 로드된 테이블이 비어있는 경우 (size == 0)
                logger.warning("로드된 선량율 테이블이 비어있습니다. GPU 전송 시도 안함.")
        except Exception as e:
            logger.error(f"__init__에서 선량율 테이블 처리 중 오류: {e}")
            self.doserate_table_gpu = None

        logger.info("스캔 시간 계산기 초기화 완료")
    
    def _load_doserate_table(self) -> Optional[np.ndarray]:
        """선량율 테이블 로드"""
        try:
            if self.doserate_table_path and os.path.exists(self.doserate_table_path):
                # 파일을 바이너리 모드로 열어서 인코딩을 확인
                with open(self.doserate_table_path, 'rb') as f:
                    # 파일의 첫 부분을 읽어서 BOM 확인
                    raw = f.read(4)
                    
                    # BOM 유무에 따라 적절한 인코딩 선택
                    encodings = ['utf-8-sig', 'utf-8', 'cp949']
                    
                    for encoding in encodings:
                        try:
                            # 파일 내용을 디코딩 시도
                            content = raw.decode(encoding)
                            # 디코딩 성공하면 해당 인코딩으로 전체 파일 로드
                            table = np.loadtxt(self.doserate_table_path, 
                                            delimiter=',',
                                            encoding=encoding)
                            logger.debug(f"선량율 테이블 로드 완료: {self.doserate_table_path} (인코딩: {encoding})")
                            return table
                        except UnicodeDecodeError:
                            continue
                        except Exception as e:
                            logger.warning(f"인코딩 {encoding}으로 파일 로드 중 오류: {e}")
                            continue
                    
                    # 모든 인코딩 시도 실패 시 에러 발생
                    raise ValueError(f"파일을 읽을 수 있는 인코딩을 찾을 수 없습니다: {self.doserate_table_path}")
            else:
                logger.warning("선량율 테이블 경로가 없거나 파일이 존재하지 않습니다. 빈 배열 반환.")
                return np.array([])
        except Exception as e:
            logger.error(f"선량율 테이블 로드 중 오류: {e}")
            return None # 오류 발생 시 None 반환
        
    def get_doserate_for_energy(self, energy: float) -> float:
        """
        에너지에 대한 선량율 반환 (캐싱 활용)
        
        Args:
            energy: 에너지 값 (MeV)
            
        Returns:
            해당 에너지에 대한 선량율 (MU/s)
        """
        # 이미 계산된 값이 있는지 확인
        if energy in self._energy_to_doserate:
            logger.debug(f"CPU 캐시에서 에너지 {energy}MeV에 대한 선량율 반환.")
            return self._energy_to_doserate[energy]

        # 사전 로드된 GPU 테이블 확인
        if self.doserate_table_gpu is not None:
            try:
                # 해당 에너지에 대한 선량율 찾기 (GPU 마스킹 연산)
                with cp.cuda.Stream(): # Ensure operations are on the correct stream if manager uses one
                    mask = (self.doserate_table_gpu[:, 0] >= energy + 0.0) & \
                           (self.doserate_table_gpu[:, 0] < energy + 0.3)
                    max_doserate_ind = cp.where(mask)[0]
                    
                    if len(max_doserate_ind) > 0:
                        # 첫 번째 매칭되는 값 사용
                        max_doserate = float(self.doserate_table_gpu[max_doserate_ind[0], 1])
                        self._energy_to_doserate[energy] = max_doserate # CPU 캐시에 저장
                        logger.debug(f"GPU 테이블 - 에너지 {energy}MeV에 대한 선량율: {max_doserate}MU/s. CPU 캐시에 저장됨.")
                        return max_doserate
                    else:
                        logger.warning(f"GPU 테이블에서 에너지 {energy}MeV에 대한 선량율을 찾을 수 없습니다. 최소 선량율 사용.")
                        # 찾지 못한 경우에도 최소 선량율을 캐시할 수 있지만, 일반적으로는 실패 케이스를 캐시하지 않음.
                        # self._energy_to_doserate[energy] = self.min_doserate
                        return self.min_doserate
            except Exception as e:
                logger.error(f"GPU 테이블에서 선량율 조회 중 오류 발생: {e}. 최소 선량율 사용.")
                # 오류 발생 시에도 최소 선량율을 캐시할 수 있음.
                # self._energy_to_doserate[energy] = self.min_doserate
                return self.min_doserate
        else:
            # GPU 테이블이 로드되지 않은 경우
            logger.warning("사전 로드된 GPU 선량율 테이블을 사용할 수 없습니다. 최소 선량율 사용.")
            # 이 경우에도 최소 선량율을 캐시할 수 있음.
            # self._energy_to_doserate[energy] = self.min_doserate
            return self.min_doserate
    
    def calculate_layer_scan_time(self, layer: Layer, DR: Optional[float] = None) -> float:
        """
        GPU에서 레이어의 스캔 시간 계산
        
        Args:
            layer: 계산할 레이어 객체
            DR: 선택적 선량율 조정값
            
        Returns:
            계산된 총 스캔 시간 (초)
        """
        try:
            # 에너지에 따른 선량율 획득
            doserate = self.get_doserate_for_energy(layer.energy)
            logger.debug(f"GPU - 레이어 에너지 {layer.energy}MeV, 선량율: {doserate}MU/s")
            
            # 라인 세그먼트 확인
            num_segments = len(layer.line_segments)
            if num_segments <= 1:
                layer.total_scan_time = 0.0
                return 0.0
            
            # 처리할 세그먼트 데이터 추출 (첫 번째 세그먼트 제외)
            segments = layer.line_segments[1:]
            num_valid_segments = len(segments)
            
            if num_valid_segments == 0:
                layer.total_scan_time = 0.0
                return 0.0
            
            # GPU 배열 준비
            with cp.cuda.Stream():
                # 거리와 무게 데이터 추출 및 GPU로 전송
                distances = np.array([seg.distance for seg in segments], dtype=np.float32)
                weights = np.array([seg.weight for seg in segments], dtype=np.float32)
                
                # GPU 배열로 변환
                dist_gpu = self.gpu_manager.array_to_gpu(distances)
                weights_gpu = self.gpu_manager.array_to_gpu(weights)
                
                if dist_gpu is None or weights_gpu is None:
                    logger.error("GPU 배열 변환 실패")
                    return 0.0
                
                # MU/cm 계산 (벡터화)
                mu_per_dist_gpu = weights_gpu / dist_gpu
                # mu_per_dist_gpu = cp.zeros_like(weights_gpu)

                # if len(dist_gpu) ~= len(dist_gpu > 0):
                #     logger.error("거리와 무게 데이터의 길이가 일치하지 않습니다")
                #     return 0.0
                
                # mask = dist_gpu > 0
                # mu_per_dist_gpu[mask] = weights_gpu[mask] / dist_gpu[mask]
                
                # 임시 선량율 계산 (벡터화)
                dose_rates_gpu = self.max_speed * mu_per_dist_gpu
                
                # 레이어의 선량율 결정
                if len(dose_rates_gpu) > 0:
                    min_dr = float(cp.min(dose_rates_gpu))
                    # 계산된 선량율과 테이블에서 가져온 선량율 중 적절한 값 사용
                    layer_doserate = max(min_dr, self.min_doserate)
                else:
                    logger.error("레이어의 선량율을 계산할 수 없습니다")
                    layer_doserate = self.min_doserate
                
                # DR 매개변수가 제공된 경우 추가 계산
                if DR is not None:
                    speeds = layer_doserate / mu_per_dist_gpu
                    DR_eff = min(DR, cp.max(speeds)/(1.2*cp.min(speeds))) 

                    max_dose_rate = float(cp.max(dose_rates_gpu))
                    layer_doserate_internal = max_dose_rate / DR_eff

                    bot_internals_gpu = weights_gpu / layer_doserate_internal

                    speed_segments_gpu = dist_gpu / bot_internals_gpu
                    speed_mask = speed_segments_gpu > self.max_speed
                    
                    # 속도 제한 초과하는 세그먼트 무게 조정
                    weights_gpu[speed_mask] = layer_doserate_internal * dist_gpu[speed_mask] / self.max_speed
                    layer_doserate = layer_doserate_internal
                
                # 가중치가 작은 세그먼트 처리
                # small_weight_mask = weights_gpu < 1e-7
                # raw_scan_time_gpu[small_weight_mask] = dist_gpu[small_weight_mask] / self.max_speed
                raw_scan_time_gpu = weights_gpu / layer_doserate
                
                # 스캔 시간 반올림
                rounded_scan_time_gpu = self.time_resolution * cp.round(raw_scan_time_gpu / self.time_resolution)
                
                # 속도 계산
                # speed_gpu = cp.zeros_like(mu_per_dist_gpu)
                # non_zero_mask = mu_per_dist_gpu > 0
                # speed_gpu[non_zero_mask] = layer_doserate / mu_per_dist_gpu[non_zero_mask]
                # speed_gpu[~non_zero_mask] = self.max_speed
                
                # 속도 및 스캔 시간 계산 준비
                # mu_per_dist_cpu = self.gpu_manager.array_to_cpu(mu_per_dist_gpu)
                # raw_scan_time_gpu = cp.zeros_like(mu_per_dist_gpu)

                # CPU로 결과 전송
                # raw_scan_time_cpu = self.gpu_manager.array_to_cpu(raw_scan_time_gpu)
                # rounded_scan_time_cpu = self.gpu_manager.array_to_cpu(rounded_scan_time_gpu)
                # speed_cpu = self.gpu_manager.array_to_cpu(speed_gpu)
                
                # if any(arr is None for arr in [mu_per_dist_cpu, rounded_scan_time_cpu]):
                #     logger.error("GPU에서 CPU로 데이터 전송 실패")
                #     return 0.0
                
                # # 세그먼트에 계산 결과 할당
                # for i, segment in enumerate(segments):
                #     segment.mu_per_dist = mu_per_dist_cpu[i]
                #     segment.rounded_scan_time = rounded_scan_time_cpu[i]
                #     segment.dose_rate = layer_doserate
                
                # 총 스캔 시간 계산
                layer.layer_doserate = layer_doserate
                layer.total_scan_time = float(cp.sum(rounded_scan_time_gpu))
                
                return layer.total_scan_time
                
        except Exception as e:
            logger.error(f"GPU 레이어 스캔 시간 계산 실패: {e}")
            return 0.0
    
    def calculate_port_scan_times(self, port: Port, DR: Optional[float] = None) -> float:
        """
        포트의 모든 레이어에 대한 GPU 기반 스캔 시간 계산
        
        Args:
            port: 계산할 포트 객체
            DR: 선택적 선량율 조정값
            
        Returns:
            포트의 총 스캔 시간 (초)
        """
        try:
            # 모든 레이어 스캔 시간을 계산
            logger.info(f"포트 처리 중 (레이어 수: {len(port.layers)})")
            
            total_scan_time = 0.0
            
            # 각 레이어 처리
            for i, layer in enumerate(port.layers):
                scan_time = self.calculate_layer_scan_time(layer, DR)
                total_scan_time += scan_time
                
                # 주기적 로깅
                if (i+1) % 10 == 0 or i+1 == len(port.layers):
                    logger.debug(f"레이어 처리 진행: {i+1}/{len(port.layers)}")
            
            # 포트 총 스캔 시간 할당
            port.total_scan_time = total_scan_time
            
            return total_scan_time
            
        except Exception as e:
            logger.error(f"포트 스캔 시간 계산 오류: {e}")
            return 0.0
    
    def calculate_all_dr_port_scan_times(self, port: Port, DR_list: List[float]) -> np.ndarray:
        """
        모든 DR 값에 대한 포트의 GPU 기반 스캔 시간 계산
        [DR × 레이어] 차원의 스캔 시간 텐서 생성
        
        Args:
            port: 계산할 포트 객체
            DR_list: DR 값 리스트
            
        Returns:
            [DR × 레이어] 차원의 스캔 시간 배열
        """
        try:
            num_dr = len(DR_list)
            num_layers = len(port.layers)
            
            logger.info(f"GPU 통합 스캔 시간 계산 시작: DR {num_dr}개 × 레이어 {num_layers}개")
            
            # 결과 배열 초기화 [DR × 레이어]
            all_scan_times = np.zeros((num_dr, num_layers), dtype=np.float32)
            
            # 각 DR에 대해 스캔 시간 계산
            for dr_idx, DR in enumerate(DR_list):
                # 포트의 모든 레이어에 대해 스캔 시간 계산
                for layer_idx, layer in enumerate(port.layers):
                    scan_time = self.calculate_layer_scan_time(layer, DR)
                    all_scan_times[dr_idx, layer_idx] = scan_time
                
                # 주기적 로깅
                if (dr_idx + 1) % 5 == 0 or dr_idx + 1 == num_dr:
                    logger.debug(f"DR 처리 진행: {dr_idx + 1}/{num_dr}")
            
            logger.info(f"GPU 통합 스캔 시간 계산 완료: {all_scan_times.shape}")
            return all_scan_times
            
        except Exception as e:
            logger.error(f"GPU 통합 스캔 시간 계산 오류: {e}")
            return np.zeros((len(DR_list), len(port.layers)), dtype=np.float32)
    
    def calculate_vectorized_dr_scan_times(self, port: Port, DR_list: List[float]) -> np.ndarray:
        """
        벡터화된 GPU 계산으로 모든 DR에 대한 스캔 시간 계산
        (추후 최적화를 위한 대안 구현)
        
        Args:
            port: 계산할 포트 객체
            DR_list: DR 값 리스트
            
        Returns:
            [DR × 레이어] 차원의 스캔 시간 배열
        """
        try:
            num_dr = len(DR_list)
            num_layers = len(port.layers)
            
            # 모든 레이어의 기본 정보 추출
            layer_energies = []
            layer_segments_data = []
            
            for layer in port.layers:
                layer_energies.append(layer.energy)
                
                # 라인 세그먼트 데이터 추출 (첫 번째 제외)
                if len(layer.line_segments) > 1:
                    segments = layer.line_segments[1:]
                    distances = [seg.distance for seg in segments]
                    weights = [seg.weight for seg in segments]
                    layer_segments_data.append((distances, weights))
                else:
                    layer_segments_data.append(([], []))
            
            # GPU로 데이터 전송 및 벡터화된 계산 수행
            # (현재는 기본 방식 사용, 추후 최적화 가능)
            return self.calculate_all_dr_port_scan_times(port, DR_list)
            
        except Exception as e:
            logger.error(f"벡터화된 스캔 시간 계산 오류: {e}")
            return self.calculate_all_dr_port_scan_times(port, DR_list)
    
    def clear_caches(self) -> None:
        """캐시 메모리 정리"""
        self._energy_to_doserate.clear()

        # GPU 선량율 테이블 메모리 해제
        if self.doserate_table_gpu is not None:
            try:
                # GPUMemoryManager를 통해 메모리 해제 요청
                # 참고: GPUMemoryManager에 특정 배열을 해제하는 기능이 없다면,
                # cupy 배열의 경우 `del self.doserate_table_gpu`가 cupy의 메모리 풀에 의해 관리될 수 있음.
                # 좀 더 명시적인 해제를 원한다면 GPUMemoryManager에 해당 기능 추가 필요.
                # 여기서는 일단 None으로 설정하여 참조를 제거하고, clean_memory()가 풀을 정리한다고 가정.
                del self.doserate_table_gpu # cupy 배열의 경우 del로 메모리 풀에 반환 시도
                self.doserate_table_gpu = None
                logger.info("GPU 선량율 테이블 메모리 참조 해제됨.")
            except Exception as e:
                logger.error(f"GPU 선량율 테이블 메모리 해제 중 오류: {e}")
                self.doserate_table_gpu = None # 오류 발생 시에도 None으로 설정

        self.gpu_manager.clean_memory()
        logger.info("모든 캐시 및 GPU 메모리 풀 정리 시도 완료.")