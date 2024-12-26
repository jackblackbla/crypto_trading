import json
import asyncio
import websockets
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
from os import makedirs
from os.path import exists

# 로깅 레벨을 INFO로 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BinanceWebSocketClient:
    def __init__(self, testnet: bool = False):
        self.base_url = (
            "wss://testnet.binancefuture.com/ws"
            if testnet else
            "wss://fstream.binance.com/ws"
        )
        
        self.ping_interval = 180
        self.pong_timeout = 600
        self.reconnect_delay = 5
        self.max_retries = 5
        
        self.current_klines: Dict[str, Dict] = {}
        self.current_orderbook: Dict[str, Dict] = {}
        makedirs('data', exist_ok=True)
        
        # 체결 데이터 카운터 (로깅 빈도 제어용)
        self.trade_counter = 0
        self.log_interval = 10  # 10건마다 로깅
    
    async def _handle_orderbook_message(self, data: Dict) -> None:
        """오더북 메시지 처리"""
        try:
            symbol = data['s'].lower()
            
            orderbook = {
                'timestamp': datetime.fromtimestamp(data['T'] / 1000),
                'symbol': symbol,
                'bid_price': float(data['b']),
                'bid_quantity': float(data['B']),
                'ask_price': float(data['a']),
                'ask_quantity': float(data['A'])
            }
            
            self.current_orderbook[symbol] = orderbook
            
            # 스프레드 계산
            spread = orderbook['ask_price'] - orderbook['bid_price']
            spread_pct = (spread / orderbook['bid_price']) * 100
            
            logging.info(
                f"[오더북] {symbol} - "
                f"매수: {orderbook['bid_price']:.1f} ({orderbook['bid_quantity']:.3f}) | "
                f"매도: {orderbook['ask_price']:.1f} ({orderbook['ask_quantity']:.3f}) | "
                f"스프레드: {spread:.1f} ({spread_pct:.3f}%)"
            )
            
        except Exception as e:
            logging.error(f"오더북 처리 중 오류: {e}")
    
    async def _handle_subscription_response(self, response: Dict) -> None:
        if 'result' in response:
            logging.info(f"구독 성공: {response}")
        elif 'error' in response:
            logging.error(f"구독 실패: {response}")
            raise Exception(f"구독 요청 실패: {response['error']}")
    
    async def _handle_trade_message(self, data: Dict) -> None:
        """체결(aggTrade) 메시지 처리"""
        try:
            trade = {
                'timestamp': datetime.fromtimestamp(data['T'] / 1000),
                'symbol': data['s'].lower(),
                'price': float(data['p']),
                'quantity': float(data['q']),
                'is_buyer_maker': data['m']
            }
            
            # 카운터 증가 및 로깅
            self.trade_counter += 1
            if self.trade_counter % self.log_interval == 0:
                direction = "매도" if trade['is_buyer_maker'] else "매수"
                logging.info(
                    f"[체결] {trade['symbol']} - "
                    f"방향: {direction}, "
                    f"가격: {trade['price']:.1f}, "
                    f"수량: {trade['quantity']:.3f}"
                )
            
        except Exception as e:
            logging.error(f"체결 처리 중 오류: {e}")
            
    async def _handle_kline_message(self, data: Dict) -> None:
        """캔들(kline) 메시지 처리"""
        try:
            k = data['k']
            symbol = data['s'].lower()
            
            candle = {
                'timestamp': datetime.fromtimestamp(k['t'] / 1000),
                'open': float(k['o']),
                'high': float(k['h']),
                'low': float(k['l']),
                'close': float(k['c']),
                'volume': float(k['v']),
                'closed': k['x']
            }
            
            if candle['closed']:
                df = pd.DataFrame([candle])
                filename = f"data/realtime_{symbol}_1m_{candle['timestamp']:%Y%m%d}.csv"
                
                df.to_csv(
                    filename,
                    mode='a',
                    header=not exists(filename),
                    index=False
                )
                logging.info(
                    f"[캔들] {symbol} - "
                    f"시가: {candle['open']:.1f}, "
                    f"종가: {candle['close']:.1f}, "
                    f"고가: {candle['high']:.1f}, "
                    f"저가: {candle['low']:.1f}, "
                    f"거래량: {candle['volume']:.3f}"
                )
            
            self.current_klines[symbol] = candle
            
        except Exception as e:
            logging.error(f"캔들 처리 중 오류: {e}")
            
    async def connect_and_subscribe(self, 
                                  symbols: List[str] = ["btcusdt"], 
                                  channels: List[str] = ["kline_1m"]) -> None:
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                streams = []
                for symbol in symbols:
                    symbol = symbol.lower()
                    for channel in channels:
                        if channel.startswith('kline'):
                            streams.append(f"{symbol}@{channel}")
                        elif channel == 'aggTrade':
                            streams.append(f"{symbol}@aggTrade")
                        elif channel == 'bookTicker':
                            streams.append(f"{symbol}@bookTicker")
                
                logging.info(f"연결 시도: {self.base_url}")
                logging.info(f"구독 스트림: {streams}")
                
                async with websockets.connect(self.base_url) as ws:
                    logging.info("WebSocket 연결 성공")
                    
                    for stream in streams:
                        subscribe_request = {
                            "method": "SUBSCRIBE",
                            "params": [stream],
                            "id": 1
                        }
                        logging.info(f"구독 요청: {subscribe_request}")
                        await ws.send(json.dumps(subscribe_request))
                    
                    while True:
                        try:
                            message = await ws.recv()
                            data = json.loads(message)
                            
                            if 'result' in data or 'error' in data:
                                await self._handle_subscription_response(data)
                                continue
                            
                            if 'data' in data:
                                data = data['data']
                            
                            if 'e' in data:
                                event_type = data['e']
                                
                                if event_type == 'kline':
                                    await self._handle_kline_message(data)
                                elif event_type == 'aggTrade':
                                    await self._handle_trade_message(data)
                                elif event_type == 'bookTicker':
                                    await self._handle_orderbook_message(data)
                            
                        except websockets.ConnectionClosed as e:
                            logging.warning(f"WebSocket 연결 종료 ({e.code} {e.reason}). 재연결...")
                            break
                        except Exception as e:
                            logging.error(f"메시지 처리 중 오류: {e}")
                            continue
                    
            except asyncio.CancelledError:
                logging.info("프로그램 종료")
                break
                
            except Exception as e:
                retry_count += 1
                logging.error(f"연결 오류 (시도 {retry_count}/{self.max_retries}): {e}")
                if retry_count < self.max_retries:
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    logging.error("최대 재시도 횟수 초과")
                    break

    def get_current_kline(self, symbol: str) -> Optional[Dict]:
        return self.current_klines.get(symbol.lower())
    
    def get_current_orderbook(self, symbol: str) -> Optional[Dict]:
        return self.current_orderbook.get(symbol.lower())

async def main():
    logging.info("Binance Futures WebSocket 클라이언트 시작")
    client = BinanceWebSocketClient(testnet=False)
    
    try:
        await client.connect_and_subscribe(
            symbols=["btcusdt"],
            channels=["kline_1m", "aggTrade", "bookTicker"]
        )
    except KeyboardInterrupt:
        logging.info("프로그램 종료")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass