import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from .base_strategy import BaseStrategy

class MLStrategy(BaseStrategy):
    def __init__(self, params: Dict[str, Any]):
        name = "ml_strategy"
        super().__init__(name, params)
        
        self.lookback = params.get('lookback', 20)
        self.predict_horizon = params.get('predict_horizon', 12)
        self.return_threshold = params.get('return_threshold', 0.005)
        self.model = None
        self.scaler = StandardScaler()
        
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """특징 엔지니어링"""
        features = pd.DataFrame(index=df.index)
        
        # 가격 변동성
        features['returns'] = df['close'].pct_change()
        features['returns_std'] = features['returns'].rolling(self.lookback).std()
        
        # 이동평균 기반 특징
        features['ema20'] = df['close'].ewm(span=20).mean()
        features['ema50'] = df['close'].ewm(span=50).mean()
        features['ema_ratio'] = features['ema20'] / features['ema50']
        features['ema_trend'] = features['ema20'].pct_change() * 100
        
        # 거래량 특징
        features['volume_ratio'] = (
            df['volume'] / df['volume'].rolling(self.lookback).mean()
        )
        features['volume_trend'] = df['volume'].pct_change() * 100
        
        # 라벨 생성 (미래 수익률)
        future_returns = (
            df['close'].shift(-self.predict_horizon) / df['close'] - 1
        ) * 100
        
        # NaN이 아닌 값에 대해서만 라벨링
        valid_returns = future_returns.dropna()
        
        # 33%, 66% 분위수 계산
        lower_threshold = valid_returns.quantile(0.33)
        upper_threshold = valid_returns.quantile(0.66)
        
        # 라벨링
        features['label'] = 0  # 기본값은 횡보(0)
        features.loc[future_returns > upper_threshold, 'label'] = 1    # 상승(1)
        features.loc[future_returns < lower_threshold, 'label'] = -1   # 하락(-1)
        
        # 분위수와 라벨 분포 출력
        self.logger.info(f"수익률 분위수: 하위 33% = {lower_threshold:.2f}%, 상위 33% = {upper_threshold:.2f}%")
        label_dist = features['label'].value_counts()
        self.logger.info(f"라벨 분포:\n{label_dist}")
        
        # NaN 처리
        features = features.fillna(0)
        
        # 특징 정보 출력
        self.logger.info("\n특징 통계:")
        self.logger.info(features.describe())
        
        return features
        
    def _train_model(self, features: pd.DataFrame) -> None:
        """모델 학습"""
        feature_cols = [col for col in features.columns if col != 'label']
        
        # 학습/검증 분할
        train_size = int(len(features) * 0.8)
        X_train = self.scaler.fit_transform(features.iloc[:train_size][feature_cols])
        y_train = features.iloc[:train_size]['label']
        
        self.model = LGBMClassifier(
            objective='multiclass',
            num_class=3,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            class_weight='balanced',
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # 학습 결과 출력
        train_pred = self.model.predict(X_train)
        from sklearn.metrics import classification_report
        self.logger.info(f"\n학습 결과:\n{classification_report(y_train, train_pred)}")
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """시그널 생성"""
        if not self.validate_data(data):
            return pd.DataFrame()
            
        features = self._prepare_features(data)
        
        # 결과 초기화
        df = data.copy()
        df['signal'] = 'HOLD'
        df['score'] = 0.0
        
        if self.model is None:
            self._train_model(features)
            
        # 예측
        feature_cols = [col for col in features.columns if col != 'label']
        X = self.scaler.transform(features[feature_cols])
        proba = self.model.predict_proba(X)
        
        # 매수/매도 시그널 생성
        df.loc[proba[:, 2] > 0.35, 'signal'] = 'BUY'
        df.loc[proba[:, 0] > 0.35, 'signal'] = 'SELL'
        
        # 신뢰도 점수
        df['score'] = (proba[:, 2] - proba[:, 0]) * 100
        
        signal_counts = df['signal'].value_counts()
        self.logger.info(
            f"ML 시그널 생성 완료: "
            f"BUY {signal_counts.get('BUY', 0)}개, "
            f"SELL {signal_counts.get('SELL', 0)}개, "
            f"HOLD {signal_counts.get('HOLD', 0)}개"
        )
        
        # 예측 확률 분포
        self.logger.info(f"\n예측 확률 분포:\n{pd.DataFrame(proba).describe()}")
        
        return df