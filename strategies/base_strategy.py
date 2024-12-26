from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional

class BaseStrategy(ABC):
    """전략 베이스 클래스"""
    
    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params
        self._setup_logging()
    
    def _setup_logging(self):
        import logging
        self.logger = logging.getLogger(f'strategy.{self.name}')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """시그널 생성 (하위 클래스에서 구현)"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """데이터 유효성 검증"""
        try:
            # 필수 컬럼 체크
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"필수 컬럼 누락: {missing_columns}")
                return False
            
            # 결측치 체크
            if data[required_columns].isnull().any().any():
                self.logger.error("데이터에 결측치(NaN) 존재")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"데이터 검증 실패: {e}")
            return False