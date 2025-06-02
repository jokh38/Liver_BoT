"""
GPU 배치 처리 유틸리티
"""
import cupy as cp
import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from tqdm import tqdm

# 로컬 모듈 임포트
from config import Config
from core.gpu_utils import GPUMemoryManager
from data_models.calculation_data import BatchData, BatchResult, TreatmentTimeResult

logger = logging.getLogger(__name__)

class BatchProcessor:
    """GPU 배치 처리 클래스"""
    
    def __init__(self, 
               num_streams: int = Config.GPU_STREAMS, 
               batch_size: int = Config.BATCH_SIZE):
        """
        배치 처리기 초기화
        
        Args:
            num_streams: 사용할 CUDA 스트림 수
            batch_size: 기본 배치 크기
        """
        self.num_streams = num_streams
        self.batch_size = batch_size
        self.gpu_manager = GPUMemoryManager()
        logger.info(f"배치 처리기 초기화 완료 (스트림: {num_streams}, 배치 크기: {batch_size})")
    
    def estimate_batch_size(self, num_combinations: int, data_size: int) -> int:
        """
        GPU 메모리 기반 최적 배치 크기 계산
        
        Args:
            num_combinations: 총 매개변수 조합 수
            data_size: 처리할 데이터 크기 (요소 수)
            
        Returns:
            최적 배치 크기
        """
        try:
            # GPU 메모리 정보 획득
            free_memory, total_memory = self.gpu_manager.get_memory_info()
            
            # 기본 메모리 요구사항 (고정 오버헤드 + 데이터)
            base_memory = 1024 * 1024 * 100  # 100MB 기본 오버헤드
            
            # 단일 데이터 항목당 메모리 (float32 = 4바이트 * 배열 크기)
            item_memory = 4 * data_size
            
            # 안전 계수 (가용 메모리의 80%만 사용)
            safety_factor = 0.8
            
            # 사용 가능한 메모리
            available_memory = int(free_memory * safety_factor) - base_memory
            
            # 배치 크기 계산
            batch_size = max(1, int(available_memory / item_memory))
            
            # 설정된 최대 배치 크기와 비교하여 최소값 선택
            optimal_batch_size = min(batch_size, self.batch_size, num_combinations)
            
            logger.info(f"메모리 분석: 가용={free_memory/1024**2:.1f}MB, 필요={item_memory*num_combinations/1024**2:.1f}MB")
            logger.info(f"최적 배치 크기: {optimal_batch_size} (전체 조합 수: {num_combinations})")
            
            return optimal_batch_size
            
        except Exception as e:
            logger.warning(f"배치 크기 계산 중 오류: {e}, 기본값 사용")
            return min(self.batch_size, num_combinations)
    
    def create_batches(self, items: List[Any], batch_size: Optional[int] = None) -> List[List[Any]]:
        """
        아이템 목록을 배치로 분할
        
        Args:
            items: 분할할 아이템 목록
            batch_size: 배치 크기 (None이면 기본값 사용)
            
        Returns:
            배치 리스트
        """
        size = batch_size or self.batch_size
        return [items[i:i+size] for i in range(0, len(items), size)]
    
    def process_batch(self, 
                    batch: List[Any], 
                    process_func: Callable, 
                    *args, 
                    **kwargs) -> List[Any]:
        """
        배치 처리 실행
        
        Args:
            batch: 처리할 배치
            process_func: 처리 함수
            *args, **kwargs: 함수에 전달할 인수
            
        Returns:
            처리 결과 리스트
        """
        streams = [cp.cuda.Stream() for _ in range(self.num_streams)]
        results = []
        
        for i, item in enumerate(batch):
            stream = streams[i % len(streams)]
            with stream:
                result = process_func(item, *args, **kwargs)
                results.append(result)
        
        # 모든 스트림 동기화
        for stream in streams:
            stream.synchronize()
            
        self.gpu_manager.clean_memory()
        return results
    
    def process_batch_data(self, 
                        batch_data: BatchData, 
                        treatment_calculator,
                        DR: float) -> BatchResult:
        """
        BatchData 객체를 이용한 통합 계산 수행
        
        Args:
            batch_data: 배치 데이터 객체 (스캔 시간, 매개변수 조합, 게이팅 주기 정보 포함)
            treatment_calculator: 치료 시간 계산기 객체
            DR: 선량율
            
        Returns:
            BatchResult: 모든 매개변수 조합에 대한 계산 결과
        """
        # 결과 객체 초기화
        batch_result = BatchResult()
        batch_result.results = {}
        
        try:
            # 스캔 시간이 없는 경우 빈 결과 반환
            if not batch_data.scan_times:
                logger.warning("처리할 스캔 시간이 없습니다.")
                return batch_result
                
            # 배치 처리를 위한 설정
            logger.info(f"총 매개변수 조합 수: {len(batch_data.parameters)}")
            data_size = len(batch_data.scan_times)
            batch_size = self.estimate_batch_size(len(batch_data.parameters), data_size)
            
            # 스캔 시간을 GPU 메모리로 전송
            scan_times_gpu = self.gpu_manager.array_to_gpu(batch_data.scan_times)
            
            if scan_times_gpu is None:
                logger.error("GPU 배열 변환 실패")
                return batch_result
            
            # 배치 처리를 위해 매개변수 분할
            param_batches = self.create_batches(batch_data.parameters, batch_size)
            
            # CUDA 스트림 생성
            num_streams = min(self.num_streams, batch_size)
            streams = [cp.cuda.Stream() for _ in range(num_streams)]
            logger.info(f"CUDA 스트림 생성 완료: {num_streams} 스트림")

            # 각 배치 처리
            for batch_idx, param_batch in enumerate(tqdm(param_batches, desc=f"DR={DR} 배치 처리", leave=False)):
                # 각 매개변수 조합에 대한 처리
                for param_idx, param_tuple in enumerate(param_batch):
                    stream_idx = param_idx % num_streams
                    stream = streams[stream_idx]
                    with stream:
                        tr, amp, tls = param_tuple
                        
                        # 게이팅 주기 정보 가져오기
                        gating_periods_gpu = batch_data.gating_periods.get((tr, amp))
                        if gating_periods_gpu is None:
                            logger.warning(f"게이팅 주기 정보 없음 (TR={tr}, amp={amp}). 계산 건너뜀.")
                            continue
                        
                        try:
                            result_dict = treatment_calculator._calculate_treatment_time_gpu(
                                scan_times           = scan_times_gpu,
                                layer_switching_time = tls,
                                gating_periods       = gating_periods_gpu
                            )
                        except Exception as e:
                            logger.error(f"치료 시간 계산 중 오류: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            continue
                        
                        # 결과 객체 생성 및 저장
                        treatment_result = TreatmentTimeResult(
                            total_time=result_dict['total_time'],
                            beam_on_time=result_dict['beam_on_time'],
                            efficiency=result_dict['efficiency']
                        )
                        
                        # 결과 저장 (CPU 메모리에 저장)
                        batch_result.results[param_tuple] = treatment_result
                
                # 각 배치 완료 후 모든 스트림 동기화
                for stream in streams:
                    stream.synchronize()
                
                # 메모리 정리 (배치 단위)
                if batch_idx % 5 == 0:
                    self.gpu_manager.clean_memory()
            
            # GPU 메모리 해제
            del scan_times_gpu
            
            # GPU 메모리 정리
            self.gpu_manager.clean_memory()
            
            logger.info(f"모든 매개변수 조합 계산 완료 (총 {len(batch_result.results)}개 결과)")
            return batch_result
            
        except Exception as e:
            logger.error(f"배치 데이터 기반 계산 중 오류 발생: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 메모리 정리 시도
            self.gpu_manager.clean_memory()
            
            return batch_result
    
    def process_full_parameter_space(self, 
                                   scan_time_tensor: np.ndarray,  # [DR × 레이어]
                                   DR_list: List[float],
                                   parameter_combinations: List[Tuple[float, float, float, float]],  # (DR, TR, amp_g, TLS)
                                   gating_periods: Dict[Tuple[float, float], 'GatingPeriod'],
                                   treatment_calculator) -> Dict[Tuple[float, float, float, float], 'TreatmentTimeResult']:
        """
        전체 매개변수 공간에 대한 GPU 배치 처리
        [DR × TR × amp_g × TLS] 전체 조합을 한 번에 계산
        
        Args:
            scan_time_tensor: [DR × 레이어] 차원의 스캔 시간 배열
            DR_list: DR 값 리스트
            parameter_combinations: 전체 매개변수 조합 리스트
            gating_periods: 게이팅 주기 정보 딕셔너리
            treatment_calculator: 치료 시간 계산기 객체
            
        Returns:
            {(DR, TR, amp_g, TLS): TreatmentTimeResult} 형태의 결과 딕셔너리
        """
        logger.info(f"전체 매개변수 공간 처리 시작: {len(parameter_combinations)}개 조합")
        
        # 결과 딕셔너리 초기화
        all_results = {}
        
        try:
            # DR 인덱스 매핑 생성
            dr_to_index = {dr: idx for idx, dr in enumerate(DR_list)}
            
            # 배치 크기 계산
            data_size = scan_time_tensor.shape[1]  # 레이어 수
            batch_size = self.estimate_batch_size(len(parameter_combinations), data_size)
            
            # 스캔 시간 텐서를 GPU로 전송 (1회)
            scan_tensor_gpu = self.gpu_manager.array_to_gpu(scan_time_tensor)
            
            if scan_tensor_gpu is None:
                logger.error("GPU 스캔 시간 텐서 전송 실패")
                return all_results
            
            logger.info(f"GPU로 스캔 시간 텐서 전송 완료: {scan_tensor_gpu.shape}")
            
            # 매개변수 조합을 배치로 분할
            param_batches = self.create_batches(parameter_combinations, batch_size)
            logger.info(f"배치 분할 완료: {len(param_batches)}개 배치, 배치당 평균 {len(parameter_combinations)//len(param_batches)}개 조합")
            
            # CUDA 스트림 생성
            num_streams = min(self.num_streams, batch_size)
            streams = [cp.cuda.Stream() for _ in range(num_streams)]
            
            # 각 배치 처리
            for batch_idx, param_batch in enumerate(tqdm(param_batches, desc="전체 매개변수 배치 처리")):
                logger.debug(f"배치 {batch_idx+1}/{len(param_batches)} 처리 중... ({len(param_batch)}개 조합)")
                
                # 각 매개변수 조합 처리
                for param_idx, (DR, TR, amp_g, TLS) in enumerate(param_batch):
                    stream_idx = param_idx % num_streams
                    stream = streams[stream_idx]
                    
                    with stream:
                        try:
                            # DR 인덱스 확인
                            dr_idx = dr_to_index.get(DR)
                            if dr_idx is None:
                                logger.warning(f"DR 값 {DR}에 대한 인덱스를 찾을 수 없습니다.")
                                continue
                            
                            # 해당 DR의 스캔 시간 추출
                            current_scan_times = scan_tensor_gpu[dr_idx, :]
                            
                            # 게이팅 주기 정보 가져오기
                            gating_period = gating_periods.get((TR, amp_g))
                            if gating_period is None:
                                logger.warning(f"게이팅 주기 정보 없음 (TR={TR}, amp_g={amp_g})")
                                continue
                            
                            # 치료 시간 계산
                            result_dict = treatment_calculator._calculate_treatment_time_gpu(
                                scan_times=current_scan_times,
                                layer_switching_time=TLS,
                                gating_periods=gating_period
                            )
                            
                            # 결과 객체 생성
                            treatment_result = TreatmentTimeResult(
                                total_time=result_dict['total_time'],
                                beam_on_time=result_dict['beam_on_time'],
                                efficiency=result_dict['efficiency']
                            )
                            
                            # 결과 저장
                            all_results[(DR, TR, amp_g, TLS)] = treatment_result
                            
                        except Exception as e:
                            logger.error(f"매개변수 조합 ({DR}, {TR}, {amp_g}, {TLS}) 계산 중 오류: {e}")
                            continue
                
                # 각 배치 완료 후 스트림 동기화
                for stream in streams:
                    stream.synchronize()
                
                # 주기적 메모리 정리
                if batch_idx % 10 == 0:
                    self.gpu_manager.clean_memory()
            
            # GPU 메모리 해제
            del scan_tensor_gpu
            self.gpu_manager.clean_memory()
            
            logger.info(f"전체 매개변수 공간 처리 완료: {len(all_results)}개 결과 생성")
            return all_results
            
        except Exception as e:
            logger.error(f"전체 매개변수 공간 처리 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 메모리 정리
            self.gpu_manager.clean_memory()
            return all_results
