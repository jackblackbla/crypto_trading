import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas_ta as ta
from tensorflow import keras
from data_collector.funding_oi_client import BinanceFundingOIClient  # 가정: data_collector 패키지 내에 정의

from .base_strategy import BaseStrategy  # BaseStrategy 클래스 import

class MLStrategy(BaseStrategy):
    def __init__(self, params: Dict[str, Any]):
        name = "ml_strategy"
        super().__init__(name, params)

        self.lookback = params.get('lookback', 20)
        self.predict_horizon = params.get('predict_horizon', 12)
        self.return_threshold = params.get('return_threshold', 0.005)
        self.model = None
        self.scaler = None
        self.model_type = params.get('model_type', 'lstm')  # 모델 타입 파라미터
        self.scaler_type = params.get('scaler_type', 'standard') # 스케일러 타입 파라미터, 기본값 설정
        self.funding_oi_client = BinanceFundingOIClient()

    def calculate_labels(self, returns: pd.Series, lower: float = 0.33, upper: float = 0.66) -> pd.Series:
        """수익률 데이터를 기반으로 라벨 생성 (0, 1, 2로 조정)"""
        labels = pd.Series(index=returns.index)
        labels[returns > returns.quantile(upper)] = 2   # High positive returns -> 2
        labels[returns < returns.quantile(lower)] = 0  # High negative returns -> 0
        labels.fillna(1, inplace=True)                  # 나머지 중립 -> 1
        return labels.astype(int)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """특징 엔지니어링"""
        features = pd.DataFrame(index=df.index)

        # 가격 변화율
        features['returns'] = df['close'].pct_change()

        # 가격 변동성
        features['returns_std'] = df['close'].pct_change().rolling(self.lookback).std()

        # 이동평균
        features['ema_50'] = df['close'].ewm(span=50).mean()
        features['ema_200'] = df['close'].ewm(span=200).mean()
        features['ema_ratio'] = features['ema_50'] / features['ema_200']
        features['ema_trend'] = features['ema_50'].pct_change() * 100

        # 볼린저 밴드
        bb = df.ta.bbands(length=20, std=2)
        features['bb_upper'] = bb['BBU_20_2.0']
        features['bb_middle'] = bb['BBM_20_2.0']
        features['bb_lower'] = bb['BBL_20_2.0']

        # RSI
        features['rsi'] = df.ta.rsi()

        # 거래량
        features['volume'] = df['volume']
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(self.lookback).mean()
        features['volume_trend'] = df['volume'].pct_change() * 100

        # 펀딩 비율 (NaN 처리 추가)
        funding_data = self.funding_oi_client.fetch_funding_rate(
            symbol="BTCUSDT",
            start_time=int(df.index.min().timestamp() * 1000),
            end_time=int(df.index.max().timestamp() * 1000)
        )
        funding_data = funding_data.set_index('timestamp')
        funding_data['fundingRate'].fillna(method='ffill', inplace=True)
        features = features.merge(funding_data[['fundingRate']], how='left', left_index=True, right_index=True)

        # OI 데이터 (NaN 처리 추가)
        oi_data = self.funding_oi_client.fetch_open_interest_hist(
            symbol="BTCUSDT",
            period="5m",
            start_time=int(df.index.min().timestamp() * 1000),
            end_time=int(df.index.max().timestamp() * 1000)
        )
        oi_data = oi_data.set_index('timestamp')
        oi_data['sumOpenInterest'].fillna(method='ffill', inplace=True)
        features = features.merge(oi_data[['sumOpenInterest']], how='left', left_index=True, right_index=True)
        features.rename(columns={'sumOpenInterest': 'oi'}, inplace=True)

        # 거래량 지표
        features['volume_ma'] = df['volume'].rolling(window=20).mean()
        features['volume_spike'] = np.where(df['volume'] > features['volume_ma'] * 2, 1, 0)

        # 변동성 지표
        features['ATR_14'] = df.ta.atr(length=14)
        features['CMO_20'] = df.ta.cmo(length=20)

        # 시간 피쳐: 인덱스가 datetime이라 가정
        features['day_of_week'] = df.index.dayofweek
        features['hour_of_day'] = df.index.hour

        # 가격 변화율
        features['close_pct_change_1'] = df['close'].pct_change(periods=1)
        features['close_pct_change_5'] = df['close'].pct_change(periods=5)

        # 지표 간 비율
        features['ema_50_200_ratio'] = features['ema_50'] / features['ema_200']

        # 이동평균의 기울기
        features['ema_50_slope'] = features['ema_50'].diff(5)

        # NaN 처리
        features.fillna(0, inplace=True)

        self.logger.info("\n특징 통계:")
        self.logger.info(features.describe())

        return features

    def _train_model(self, X_train, y_train, X_val, y_val) -> None:
        """모델 학습 (TensorFlow/Keras 버전)"""
        if self.model_type == 'lstm':
            X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_val_reshaped = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
            self.model = keras.Sequential([
                keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], 1)),
                keras.layers.LSTM(units=50),
                keras.layers.Dense(3, activation='softmax')
            ])
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            self.model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_val_reshaped, y_val), verbose=0)

        elif self.model_type == 'cnn':
            X_train_cnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_val_cnn = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
            self.model = keras.Sequential([
                keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
                keras.layers.MaxPooling1D(pool_size=2),
                keras.layers.Flatten(),
                keras.layers.Dense(50, activation='relu'),
                keras.layers.Dense(3, activation='softmax')
            ])
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            self.model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_data=(X_val_cnn, y_val), verbose=0)

        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
        
        # 검증 데이터로 성능 평가
        if self.model_type in ['lstm', 'cnn']:
            if self.model_type == 'lstm':
                X_val_reshaped = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
            else:
                X_val_reshaped = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
            val_loss, val_accuracy = self.model.evaluate(X_val_reshaped, y_val, verbose=0)
            self.logger.info(f"검증 데이터에 대한 손실: {val_loss:.4f}, 정확도: {val_accuracy:.4f}")
            
            # 검증 데이터에 대한 예측 수행 및 평가
            val_predictions = np.argmax(self.model.predict(X_val_reshaped), axis=1)
            self.logger.info(f"\n검증 결과:\n{classification_report(y_val, val_predictions)}")
        else:
            val_predictions = self.model.predict(X_val)
            self.logger.info(f"\n검증 결과:\n{classification_report(y_val, val_predictions)}")
            
        
        # 훈련 결과 로깅
        train_pred = np.argmax(self.model.predict(X_train_reshaped if self.model_type in ['lstm', 'cnn'] else X_train), axis=-1)
        self.logger.info(f"\n학습 결과 ({self.model_type}):\n{classification_report(y_train, train_pred)}")
        
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """시그널 생성 (TensorFlow/Keras 버전)"""
        if not self.validate_data(df):
            return pd.DataFrame()

        # 특징 생성
        features = self._prepare_features(df)
        features.dropna(inplace=True)

        # 라벨 생성
        labels = self.calculate_labels(features['returns']).reindex(features.index)
        labels.dropna(inplace=True)

        # 스케일러 선택
        if self.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        self.scaler = scaler

        # 사용할 feature 목록
        feature_cols = [
            "returns", "returns_std", "ema_50", "ema_200", "ema_ratio", "ema_trend",
            "volume_ratio", "volume_trend", "ATR_14", "CMO_20", "volume_ma", "volume_spike",
            "day_of_week", "hour_of_day", "close_pct_change_1", "close_pct_change_5",
            "ema_50_200_ratio", "ema_50_slope"
        ]

        # 오버샘플링 (선택)
        oversampler = RandomOverSampler(random_state=42)
        X_oversampled, y_oversampled = oversampler.fit_resample(
            features[feature_cols],
            labels
        )
        self.logger.info("[INFO] Label Distribution (after oversampling):")
        self.logger.info(pd.Series(y_oversampled).value_counts())

        # 훈련 데이터 준비
        X_train = self.scaler.fit_transform(X_oversampled)
        y_train = y_oversampled

        # 검증 데이터 준비 (최근 20%를 사용)
        val_size = int(len(X_train) * 0.2)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]

        # 모델이 없으면 훈련
        if self.model is None:
            self._train_model(X_train, y_train, X_val, y_val)

        # 예측
        df_signals = df.copy()
        df_signals['signal'] = 'HOLD'
        df_signals['score'] = 0.0

        try:
            X = self.scaler.transform(features[feature_cols])

            # 모델 타입에 따른 예측
            if self.model_type == 'lstm':
                X_reshaped = np.reshape(X, (X.shape[0], X.shape[1], 1))
                predictions = np.argmax(self.model.predict(X_reshaped, verbose=0), axis=-1)
                proba = self.model.predict(X_reshaped, verbose=0)
            elif self.model_type == 'cnn':
                X_reshaped = np.reshape(X, (X.shape[0], X.shape[1], 1))
                predictions = np.argmax(self.model.predict(X_reshaped, verbose=0), axis=-1)
                proba = self.model.predict(X_reshaped, verbose=0)
            else:
                predictions = self.model.predict(X)
                proba = self.model.predict_proba(X)

            # 시그널 및 점수 저장
            df_signals['prediction'] = predictions
            df_signals['signal'] = df_signals['prediction'].apply(lambda x: 'BUY' if x == 2 else 'SELL' if x == 0 else 'HOLD')
            df_signals['score'] = np.max(proba, axis=1)

            # 결과 로깅
            signal_counts = df_signals['signal'].value_counts()
            self.logger.info(f"ML 시그널 생성 완료: BUY {signal_counts.get('BUY', 0)}개, SELL {signal_counts.get('SELL', 0)}개, HOLD {signal_counts.get('HOLD', 0)}개")
            self.logger.info(f"\n예측 확률 분포:\n{pd.DataFrame(proba).describe()}")

            return df_signals

        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return pd.DataFrame()