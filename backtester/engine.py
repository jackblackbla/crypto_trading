# backtester/engine.py
import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(
        self, 
        strategy,          # 전략 객체 (MLStrategy 등)
        initial_balance=10000.0, 
        fee_rate=0.0005,   # 예: 0.05% 수수료
        slippage=0.0005,   # 예: 0.05% 슬리피지
    ):
        """
        간단한 현물 매매 백테스트 엔진
        """
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage

        # 결과 저장
        self.trades = []    # 매매 내역(진입, 청산 등)
        self.equity_curve = []  # 시점별 잔고

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df: 시뮬레이션에 사용할 데이터 (OHLCV, etc.)
        return: 결과(매 시점 포지션/잔고 등) DataFrame
        """
        # step 1) 시그널 생성 (MLStrategy or 다른 strategy.generate_signals)
        result_df = self.strategy.generate_signals(df)

        # 백테스트용 컬럼 추가
        result_df['position'] = 'FLAT'   # 보유 상태: FLAT(미보유), LONG(매수 상태) 가정
        result_df['hold_amount'] = 0.0   # 코인 보유 수량
        result_df['balance'] = self.initial_balance  # 현금(USDT or KRW) 잔고
        result_df['pnl'] = 0.0           # 해당 시점에서의 실현 손익

        position = 'FLAT'
        entry_price = 0.0
        hold_amount = 0.0
        balance = self.initial_balance

        # step 2) 시점 순회
        for i in range(len(result_df)):
            signal = result_df['signal'].iloc[i]
            close_price = result_df['close'].iloc[i]

            # 슬리피지 반영
            # ex) 매수 시 약간 높은 가격으로 체결, 매도 시 약간 낮은 가격으로 체결
            fill_price = close_price * (1 + self.slippage) if signal == 'BUY' else \
                         close_price * (1 - self.slippage) if signal == 'SELL' else close_price

            if position == 'FLAT':
                # 매수 시그널 -> 진입
                if signal == 'BUY':
                    # 수수료 계산
                    amount_to_use = balance * (1 - self.fee_rate)
                    hold_amount = amount_to_use / fill_price
                    balance_after = balance - amount_to_use  # 현금은 거의 0에 가깝게
                    position = 'LONG'
                    entry_price = fill_price
                    # 트레이드 기록
                    self.trades.append({
                        'time': result_df.index[i],
                        'type': 'BUY',
                        'price': fill_price,
                        'size': hold_amount,
                        'balance_before': balance,
                        'balance_after': balance_after
                    })
                    balance = balance_after

            elif position == 'LONG':
                # 매도 시그널 -> 청산
                if signal == 'SELL':
                    # 매도 체결
                    proceeds = hold_amount * fill_price
                    fee = proceeds * self.fee_rate
                    proceeds_after_fee = proceeds - fee
                    # 실현 PnL = (체결금액 - 매수금액)
                    realized_pnl = proceeds_after_fee - (hold_amount * entry_price)
                    balance_after = balance + proceeds_after_fee
                    # 트레이드 기록
                    self.trades.append({
                        'time': result_df.index[i],
                        'type': 'SELL',
                        'price': fill_price,
                        'size': hold_amount,
                        'balance_before': balance,
                        'balance_after': balance_after,
                        'realized_pnl': realized_pnl
                    })
                    position = 'FLAT'
                    hold_amount = 0.0
                    entry_price = 0.0
                    balance = balance_after

            # 저장
            result_df.loc[result_df.index[i], 'position'] = position
            result_df.loc[result_df.index[i], 'hold_amount'] = hold_amount
            result_df.loc[result_df.index[i], 'balance'] = balance

            # 실현 PnL은 해당 봉에서만 발생(매도 시)
            if signal == 'SELL' and position == 'FLAT':
                result_df.loc[result_df.index[i], 'pnl'] = realized_pnl

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