import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(
        self, 
        strategy,          # 전략 객체 (MLStrategy 등)
        initial_balance=10000.0,
        fee_rate=0.0005,   # 예: 0.05% 수수료
        slippage=0.0005,   # 예: 0.05% 슬리피지
        risk_per_trade=0.01, # Adjust as needed (e.g., 0.01 for 1% risk)
    ):
        """
        간단한 현물 매매 백테스트 엔진, 포지션 사이징 기능 추가
        """
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade

        # 백테스트 결과 저장
        self.trades = []    # 매매 기록 (진입, 청산 등)
        self.equity_curve = []  # 시점별 잔고

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df: 시뮬레이션에 사용할 데이터 (OHLCV, etc.)
        return: 결과(매 시점 포지션/잔고 등) DataFrame
        """
        # step 1) 시그널 생성 (MLStrategy or 다른 strategy.generate_signals)
        result_df = self.strategy.generate_signals(df)

        # 시그널 생성 결과가 비어 있는지 확인
        if result_df.empty:
            print("[Warning] generate_signals returned an empty DataFrame. Skipping backtest.")
            return pd.DataFrame()

        # 백테스트용 컬럼 추가
        result_df['position'] = 'FLAT'   # 보유 상태: FLAT(미보유), LONG(매수 상태) 가정
        result_df['hold_amount'] = 0.0   # 코인 보유 수량
        result_df['balance'] = pd.Series(self.initial_balance, dtype=float)  # 현금(USDT or KRW) 잔고, float 타입으로 변경
        result_df['pnl'] = 0.0           # 해당 시점에서의 실현 손익

        position = 'FLAT'
        entry_price = 0.0
        hold_amount = 0.0
        balance = float(self.initial_balance)
        realized_pnl = 0.0
        
        # 분할 매수/매도 횟수 및 비율 설정
        self.buy_split = 3
        self.sell_split = 3
        self.buy_count = 0
        self.sell_count = 0

        # 손절 및 익절 비율 설정
        self.stop_loss_rate = 0.05
        self.take_profit_rate = 0.10
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0

        # step 2) 시점 순회
        for i in range(len(result_df)):
            signal = result_df['signal'].iloc[i]
            close_price = result_df['close'].iloc[i]
            volume = result_df['volume'].iloc[i]

            # 슬리피지 반영 (가변 슬리피지)
            slippage_factor = 0.0005  # 기본 슬리피지 0.05%
            if volume < 100:
                slippage_factor = 0.001  # 거래량 적을 시 슬리피지 0.1%
            fill_price = close_price * (1 + slippage_factor) if signal == 'BUY' else \
                         close_price * (1 - slippage_factor) if signal == 'SELL' else close_price

            if position == 'FLAT':
                # 매수 시그널 -> 분할 진입
                if signal == 'BUY' and self.buy_count < self.buy_split:
                    print(f"[DEBUG] time={result_df.index[i]}, BUY, fill_price={fill_price}, balance_before={balance}")
                    
                    # 수수료를 먼저 제외하고 계산 (업비트 현물 수수료율 0.05%)
                    fee_rate = 0.0005
                    amount_to_use = balance / (self.buy_split - self.buy_count)
                    fee = amount_to_use * fee_rate
                    amount_to_use_after_fee = amount_to_use - fee
                    
                    buy_amount = amount_to_use_after_fee / fill_price
                    balance_after = balance - amount_to_use
                    
                    self.trades.append({
                        'time': result_df.index[i],
                        'type': 'BUY',
                        'price': fill_price,
                        'size': buy_amount,
                        'balance_before': balance,
                        'balance_after': balance_after,
                        'fee': fee
                    })
                    
                    balance = balance_after
                    hold_amount += buy_amount
                    self.buy_count += 1
                    entry_price = (entry_price * (self.buy_count - 1) + fill_price) / self.buy_count
                    position = 'LONG'
                    self.stop_loss_price = fill_price * (1 - self.stop_loss_rate)
                    self.take_profit_price = fill_price * (1 + self.take_profit_rate)

            elif position == 'LONG':
                # 펀딩비 반영 (스윙 전략에만 적용)
                if 'funding_rate' in result_df.columns:
                    funding_payment = hold_amount * result_df['funding_rate'].iloc[i] * entry_price
                    if funding_payment > 0:
                        print(f"[DEBUG] time={result_df.index[i]}, 펀딩비 지불: {funding_payment}")
                    elif funding_payment < 0:
                        print(f"[DEBUG] time={result_df.index[i]}, 펀딩비 수취: {funding_payment}")
                    balance -= funding_payment

                # 손절/익절 로직
                if close_price <= self.stop_loss_price or close_price >= self.take_profit_price:
                    print(f"[DEBUG] time={result_df.index[i]}, 손절/익절, fill_price={fill_price}, balance_before={balance}")
                    # 전량 청산
                    proceeds = hold_amount * fill_price
                    fee = proceeds * self.fee_rate
                    proceeds_after_fee = proceeds - fee
                    realized_pnl = proceeds_after_fee - (hold_amount * entry_price)
                    balance_after = balance + proceeds_after_fee

                    self.trades.append({
                        'time': result_df.index[i],
                        'type': 'SELL',
                        'price': fill_price,
                        'size': hold_amount,
                        'balance_before': balance,
                        'balance_after': balance_after,
                        'realized_pnl': realized_pnl,
                        'fee': fee
                    })

                    balance = balance_after
                    position = 'FLAT'
                    hold_amount = 0.0
                    entry_price = 0.0
                    self.buy_count = 0
                    self.sell_count = 0
                    self.stop_loss_price = 0.0
                    self.take_profit_price = 0.0

                # 매도 시그널 -> 분할 청산
                elif signal == 'SELL' and self.sell_count < self.sell_split:
                    print(f"[DEBUG] time={result_df.index[i]}, SELL, fill_price={fill_price}, balance_before={balance}")
                    sell_amount = hold_amount / (self.sell_split - self.sell_count)
                    proceeds = sell_amount * fill_price
                    fee = proceeds * self.fee_rate
                    proceeds_after_fee = proceeds - fee
                    realized_pnl = proceeds_after_fee - (sell_amount * entry_price)
                    balance_after = balance + proceeds_after_fee

                    self.trades.append({
                        'time': result_df.index[i],
                        'type': 'SELL',
                        'price': fill_price,
                        'size': sell_amount,
                        'balance_before': balance,
                        'balance_after': balance_after,
                        'realized_pnl': realized_pnl,
                        'fee': fee
                    })

                    balance = balance_after
                    hold_amount -= sell_amount
                    self.sell_count += 1

                    # 모두 청산했으면 position을 FLAT으로 변경
                    if hold_amount < 0.000001:
                        position = 'FLAT'
                        entry_price = 0.0
                        self.buy_count = 0
                        self.sell_count = 0
                        self.stop_loss_price = 0.0
                        self.take_profit_price = 0.0

            # 저장 (iloc 사용으로 변경)
            result_df.iloc[i, result_df.columns.get_loc('position')] = position
            result_df.iloc[i, result_df.columns.get_loc('hold_amount')] = hold_amount
            result_df.iloc[i, result_df.columns.get_loc('balance')] = balance

            # 실현 PnL은 해당 봉에서만 발생(매도 시)
            if signal == 'SELL':
                if position == 'LONG':
                    result_df.iloc[i, result_df.columns.get_loc('pnl')] = realized_pnl
                else:
                    result_df.iloc[i, result_df.columns.get_loc('pnl')] = 0

            # or 만약 매봉마다 "미실현 손익"을 equity_curve로 기록하고 싶으면:
            if position == 'LONG':
                # 미실현 PnL: (현재가격 - 진입가)*수량 (수수료 제외)
                unrealized_pnl = (close_price - entry_price) * hold_amount
                cur_equity = balance + unrealized_pnl
            else:
                # FLAT이면 그냥 balance = total equity
                cur_equity = balance

            self.equity_curve.append(cur_equity)

        return result_df

    def summary(self):
        """
        백테스트 결과 요약 출력
        """
        if not self.trades:
            print("No trades executed.")
            return

        # 최종 잔고
        final_balance = self.equity_curve[-1]
        total_return = (final_balance - self.initial_balance) / self.initial_balance * 100

        # 최대 낙폭(MDD) 계산
        peak = -9999999999
        drawdown = 0
        mdd = 0
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak * 100
            if dd < drawdown:
                drawdown = dd
        mdd = abs(drawdown)

        print(f"Final Balance: {final_balance:.2f}, Total Return: {total_return:.2f}%")
        print(f"Max Drawdown: {mdd:.2f}%")
        print(f"Number of trades: {len(self.trades)}")

        # 간단하게 트레이드 내역도 일부 표시
        df_trades = pd.DataFrame(self.trades)
        buy_count = len(df_trades[df_trades['type'] == 'BUY'])
        sell_count = len(df_trades[df_trades['type'] == 'SELL'])
        print(f"Buy Trades: {buy_count}, Sell Trades: {sell_count}")
        # 추가 분석(승률, 평균수익 등)은 자유롭게 계산 가능
