"""
DICOM 파일 로드 및 처리 모듈
"""
import numpy as np
import pydicom
import logging
import pathlib
import os
from typing import Optional, List, Dict, Tuple

# 로컬 모듈 임포트
from config import Config, DOSERATE_TABLE_PATH
from data_models.dicom_structures import LineSegment, Layer, Port

# 로깅 설정
logger = logging.getLogger(__name__)

class DicomLoader:
    """Dicom RT 이온 계획 파일에서 위치와 가중치 정보를 추출하는 클래스"""
    def __init__(self, file_name: Optional[str] = None, debug_mode: bool = False) -> None:
        # 공개 속성 초기화
        self.ports = []           # 포트(빔) 정보 리스트
        self.debug_mode = debug_mode  # 디버깅 모드 여부
        
        # 숨겨진 속성 초기화
        self.RTion_name = file_name
        
        # 파일이 제공된 경우 바로 처리 시작
        if file_name is not None:
            self.load_dicom_file()
    
    def load_dicom(self, file_path: str) -> Optional[pydicom.dataset.FileDataset]:
        """DICOM 파일 로드
        
        Args:
            file_path: DICOM 파일 경로
            
        Returns:
            로드된 DICOM 데이터, 실패 시 None
        """
        try:
            self.RTion_name = file_path
            return pydicom.dcmread(file_path)
        except Exception as e:
            logger.error(f"DICOM 파일 로드 오류: {str(e)}")
            return None
    
    def load_dicom_file(self) -> None:
        """Dicom 파일을 처리하여 포트 및 레이어 정보 추출"""
        try:
            # Dicom 파일 읽기 - load_dicom 메서드 사용
            d_header = self.load_dicom(self.RTion_name)
            if d_header is None:
                logger.error("DICOM 파일을 로드할 수 없습니다.")
                self.ports = []  # 빈 리스트로 초기화
                return
                
            # IonBeamSequence 속성이 있는지 확인
            if not hasattr(d_header, 'IonBeamSequence') or not d_header.IonBeamSequence:
                logger.error("DICOM 파일에 IonBeamSequence가 없거나 비어 있습니다.")
                self.ports = []  # 빈 리스트로 초기화
                return
                
            beam_sequence = list(d_header.IonBeamSequence)
                        
            # 각 포트(빔) 처리
            for i_port, port_name in enumerate(beam_sequence):
                port = Port()
                
                info_layer = port_name.IonControlPointSequence
                layers_info = list(info_layer)
                N_layers = len(layers_info) // 2
                
                # MLC position for each port
                if hasattr(port_name, 'IonBeamLimitingDeviceSequence'):
                    port.mlc_y = list(port_name.IonBeamLimitingDeviceSequence[0][0x300a, 0x00be])
                    
                # Aperture information for each port
                if int(port_name.NumberOfBlocks) > 0:
                     aperture_data = 0.1*np.array(port_name.IonBlockSequence[0][0x300a, 0x0106].value, dtype=float)
                     port.aperture = np.reshape(aperture_data, (len(aperture_data)//2, 2))
                else:
                    port.aperture = None
                                
                # 각 레이어 처리
                for i_layer in range(N_layers):
                    jj = 2 * i_layer
                    
                    # 에너지 및 누적 가중치 정보
                    energy = layers_info[jj].NominalBeamEnergy
                    cum_weight_now = layers_info[jj].CumulativeMetersetWeight
                    cum_weight_next = layers_info[jj+1].CumulativeMetersetWeight
                    
                    # 라인 스캔 위치 맵
                    position_data = np.frombuffer(layers_info[jj][0x300b, 0x1094].value, dtype=np.float32)
                    positions = np.reshape(0.1*position_data, (len(position_data)//2, 2))
                    
                    # 라인 스캔 가중치
                    weights = np.frombuffer(layers_info[jj][0x300b, 0x1096].value, dtype=np.float32)
                    
                    # MLC positions (cm)
                    if (0x300a, 0x011a) in layers_info[jj]:
                        mlc_sequence = layers_info[jj].get((0x300a, 0x011a))
                        if mlc_sequence:
                            mlc_position_data = 0.1 * np.array(mlc_sequence[0][0x300a, 0x011c].value, dtype=float)
                            mlc_pos = np.reshape(mlc_position_data, (len(mlc_position_data)//2, 2), order='F')
                        else:
                            mlc_pos = None
                    else:
                        mlc_pos = None
                    
                    # 기타 정보
                    num_positions = layers_info[jj][0x300b, 0x1092].value.decode('ascii').strip()
                    tune_id = layers_info[jj][0x300b, 0x1090].value.decode('ascii').strip()
                                        
                    # 레이어 객체 생성
                    layer = Layer(
                        energy=energy,
                        cum_weight_now=cum_weight_now,
                        cum_weight_next=cum_weight_next,
                        positions=positions,
                        weights=weights,
                        mlc_positions=mlc_pos,
                        num_positions=num_positions,
                        tune_id=tune_id
                    )
                    
                    # 레이어 정보 저장
                    port.layers.append(layer)
                    
                    # 임시 변수 명시적 해제
                    del positions, weights, position_data
                    
                # 포트 정보 저장                
                self.ports.append(port)
                
            # 대용량 변수 명시적 해제
            del d_header
            del beam_sequence
            
            # 메모리 자원 정리
            self.cleanup_resources()
                
        except pydicom.errors.InvalidDicomError:
            logger.error("유효하지 않은 DICOM 파일입니다.")
        except FileNotFoundError:
            logger.error(f"파일을 찾을 수 없습니다: {self.RTion_name}")
        except Exception as e:
            logger.exception(f"Dicom 처리 중 오류 발생: {e}")
    
    # 메모리 리소스 정리가 필요한 경우 아래 메소드를 호출하세요
    def cleanup_resources(self) -> None:
        """메모리 사용 최적화를 위한 정리 작업 수행"""
        # 디버그 모드 여부와 관계없이 항상 메모리 정리 수행
        import gc
        gc.collect()
        logger.debug("메모리 자원 정리 완료")
            
    def get_ports(self) -> List[Port]:
        """포트 목록 반환"""
        return self.ports
