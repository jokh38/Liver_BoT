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
    
    def estimate_batch_size(self, num_combinations: int, main_data_gpu_size_bytes: int = 0, per_item_additional_gpu_bytes: int = 1024*100) -> int:
        """
        GPU 메모리 기반 최적 배치 크기 동적 계산
        
        Args:
            num_combinations: 총 처리해야 할 아이템(조합) 수
            main_data_gpu_size_bytes: 현재 GPU에 로드된 주 데이터의 크기 (바이트 단위)
            per_item_additional_gpu_bytes: 각 아이템(조합)을 처리하는 데 필요한 추가 GPU 메모리 (바이트 단위)
                                           (예: 각 CUDA 태스크에 필요한 임시 버퍼 등)
            
        Returns:
            계산된 최적 배치 크기
        """
        try:
            free_memory, total_memory = self.gpu_manager.get_memory_info()
            if free_memory is None or total_memory is None: # GPU 정보 조회 실패 시
                logger.warning("GPU 메모리 정보 조회 실패. 기본 배치 크기 사용.")
                return min(self.batch_size, num_combinations)

            base_overall_overhead = 1024 * 1024 * 100  # CuPy 컨텍스트, 일반 버퍼 등을 위한 100MB 기본 오버헤드
            safety_factor = 0.8  # 가용 메모리의 80% 사용

            # 배치 아이템 처리를 위해 사용 가능한 메모리
            usable_memory_for_batch_items = (free_memory * safety_factor) - base_overall_overhead - main_data_gpu_size_bytes

            logger.debug(f"GPU 메모리: 총 {total_memory / (1024**2):.1f}MB, "
                         f"가용 {free_memory / (1024**2):.1f}MB, "
                         f"안전계수 적용 가용: {(free_memory * safety_factor) / (1024**2):.1f}MB")
            logger.debug(f"메모리 계산: 기본 오버헤드 {base_overall_overhead / (1024**2):.1f}MB, "
                         f"주 데이터 크기 {main_data_gpu_size_bytes / (1024**2):.1f}MB")
            logger.debug(f"배치 아이템 가용 메모리: {usable_memory_for_batch_items / (1024**2):.1f}MB")

            if usable_memory_for_batch_items <= 0:
                logger.warning(f"배치 아이템 처리에 사용 가능한 GPU 메모리 부족 "
                               f"({usable_memory_for_batch_items / (1024**2):.1f}MB). "
                               f"배치 크기를 1로 설정합니다.")
                return 1

            if per_item_additional_gpu_bytes <= 0: # 0 또는 음수일 경우, 아이템당 메모리 제한 없음
                calculated_batch_size = num_combinations
                logger.info(f"per_item_additional_gpu_bytes가 0이거나 작으므로, "
                            f"아이템당 메모리 제한 없이 배치 크기 설정: {calculated_batch_size}")
            else:
                calculated_batch_size = max(1, int(usable_memory_for_batch_items / per_item_additional_gpu_bytes))
                logger.info(f"아이템당 필요 메모리 {per_item_additional_gpu_bytes / 1024:.1f}KB 기준, "
                            f"계산된 배치 크기: {calculated_batch_size}")

            # Config의 BATCH_SIZE (최대 허용 배치 크기)와 num_combinations (총 아이템 수)를 넘지 않도록 조정
            optimal_batch_size = min(calculated_batch_size, self.batch_size, num_combinations)
            
            logger.info(f"최종 최적 배치 크기: {optimal_batch_size} "
                        f"(요청된 조합 수: {num_combinations}, 최대 배치 크기 설정: {self.batch_size})")
            
            return optimal_batch_size
            
        except Exception as e:
            logger.error(f"배치 크기 계산 중 예외 발생: {e}. 기본 배치 크기 사용.")
            import traceback
            logger.error(traceback.format_exc())
            return min(self.batch_size, num_combinations) # 예외 발생 시 안전한 기본값 반환
    
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
            
            # 스캔 시간을 GPU 메모리로 전송
            scan_times_gpu = self.gpu_manager.array_to_gpu(batch_data.scan_times)
            
            if scan_times_gpu is None:
                logger.error("GPU 배열 변환 실패 (scan_times_gpu)")
                # scan_times_gpu가 None이면 .nbytes 접근 시 오류 발생하므로 여기서 처리
                current_main_data_size = 0
                # 이 경우, estimate_batch_size가 메모리 부족으로 1을 반환할 가능성이 높음
                # 또는 여기서 처리를 중단하고 빈 결과를 반환할 수도 있음
            else:
                current_main_data_size = scan_times_gpu.nbytes

            # 배치 크기 동적 계산
            # 각 파라미터 조합 처리 시 scan_times_gpu (주 데이터)는 공유되고,
            # 각 태스크(조합)당 추가적인 작은 메모리(예: 결과 저장, 임시 변수 등)가 필요하다고 가정
            # 여기서는 100KB로 설정 (per_item_additional_gpu_bytes)
            batch_size = self.estimate_batch_size(
                num_combinations=len(batch_data.parameters),
                main_data_gpu_size_bytes=current_main_data_size,
                per_item_additional_gpu_bytes=1024*100  # 예: 100KB
            )
            if batch_size == 1 and current_main_data_size > (self.gpu_manager.get_memory_info()[0] * 0.5): # 매우 큰 주 데이터로 배치1이 된 경우
                logger.warning("배치 크기가 1로 계산되었으며, 주 데이터가 GPU 메모리의 상당 부분을 차지합니다. "
                               "메모리 부족 위험이 있을 수 있습니다.")

            if scan_times_gpu is None and len(batch_data.parameters) > 0 : # 재확인: scan_times_gpu가 없고 처리할 파라미터가 있다면
                logger.error("scan_times_gpu가 GPU로 전송되지 않았으나 처리할 파라미터가 있습니다. 진행 중단.")
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
            
            # 스캔 시간 텐서를 GPU로 전송 (1회)
            scan_tensor_gpu = self.gpu_manager.array_to_gpu(scan_time_tensor)

            if scan_tensor_gpu is None:
                logger.error("GPU 스캔 시간 텐서 전송 실패 (scan_tensor_gpu)")
                current_main_data_size = 0
            else:
                current_main_data_size = scan_tensor_gpu.nbytes
                logger.info(f"GPU로 스캔 시간 텐서 전송 완료: {scan_tensor_gpu.shape}, 크기: {current_main_data_size / (1024**2):.2f} MB")

            # 배치 크기 동적 계산
            # scan_tensor_gpu (주 데이터)는 공유, 각 파라미터 조합 처리에 추가 메모리 필요 가정
            batch_size = self.estimate_batch_size(
                num_combinations=len(parameter_combinations),
                main_data_gpu_size_bytes=current_main_data_size,
                per_item_additional_gpu_bytes=1024*100  # 예: 100KB, 각 조합 처리 오버헤드
            )
            if batch_size == 1 and current_main_data_size > (self.gpu_manager.get_memory_info()[0] * 0.5): # 매우 큰 주 데이터로 배치1이 된 경우
                logger.warning("배치 크기가 1로 계산되었으며, 주 데이터가 GPU 메모리의 상당 부분을 차지합니다. "
                               "메모리 부족 위험이 있을 수 있습니다.")

            if scan_tensor_gpu is None and len(parameter_combinations) > 0:
                logger.error("scan_tensor_gpu가 GPU로 전송되지 않았으나 처리할 파라미터 조합이 있습니다. 진행 중단.")
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
