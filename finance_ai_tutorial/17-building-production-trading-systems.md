# Module 17: Building Production Trading Systems

## Table of Contents
1. [System Architecture Design](#system-architecture-design)
2. [Real-Time Data Processing](#real-time-data-processing)
3. [Low-Latency Infrastructure](#low-latency-infrastructure)
4. [Order Management Systems](#order-management-systems)
5. [Monitoring and Alerting](#monitoring-and-alerting)
6. [Testing and Deployment](#testing-and-deployment)
7. [PhD-Level Research Topics](#phd-level-research-topics)

## System Architecture Design

### Microservices Architecture

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import aiohttp

class OrderRequest(BaseModel):
    symbol: str
    side: str
    quantity: int
    order_type: str
    price: Optional[float] = None

class TradingAPIService:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.app = FastAPI()
        self.host = host
        self.port = port
        self._setup_routes()
        
    def _setup_routes(self):
        @self.app.post("/order")
        async def place_order(order: OrderRequest):
            try:
                result = await self._process_order(order)
                return {"status": "success", "order_id": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/position/{symbol}")
        async def get_position(symbol: str):
            position = await self._get_position(symbol)
            return {"symbol": symbol, "position": position}
        
        @self.app.get("/pnl")
        async def get_pnl():
            pnl = await self._calculate_pnl()
            return {"pnl": pnl}
    
    async def _process_order(self, order: OrderRequest) -> str:
        import uuid
        
        order_id = str(uuid.uuid4())
        
        return order_id
    
    async def _get_position(self, symbol: str) -> Dict:
        return {"quantity": 0, "avg_price": 0.0}
    
    async def _calculate_pnl(self) -> float:
        return 0.0
    
    def run(self):
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)
```

### Event-Driven Architecture

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio
from typing import Callable, List

class EventType(Enum):
    MARKET_DATA = "market_data"
    ORDER = "order"
    FILL = "fill"
    RISK = "risk"
    POSITION = "position"

@dataclass
class Event:
    event_type: EventType
    timestamp: datetime
    data: Dict

class EventBus:
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_queue = asyncio.Queue()
        
    def subscribe(self, event_type: EventType, handler: Callable):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(handler)
    
    async def publish(self, event: Event):
        await self.event_queue.put(event)
    
    async def process_events(self):
        while True:
            event = await self.event_queue.get()
            
            handlers = self.subscribers.get(event.event_type, [])
            
            tasks = [handler(event) for handler in handlers]
            
            await asyncio.gather(*tasks)


class MarketDataHandler:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        
        event_bus.subscribe(EventType.MARKET_DATA, self.handle_market_data)
    
    async def handle_market_data(self, event: Event):
        symbol = event.data['symbol']
        price = event.data['price']
        
        print(f"Processing market data: {symbol} @ {price}")
```

## Real-Time Data Processing

### Apache Kafka Integration

```python
from kafka import KafkaProducer, KafkaConsumer
import json
from typing import Dict, Callable

class MarketDataStream:
    def __init__(self, bootstrap_servers: List[str]):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
    def publish_market_data(self, topic: str, symbol: str, data: Dict):
        message = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        self.producer.send(topic, value=message)
        self.producer.flush()
    
    def subscribe_to_market_data(
        self,
        topics: List[str],
        callback: Callable[[Dict], None]
    ):
        self.consumer.subscribe(topics)
        
        for message in self.consumer:
            callback(message.value)


class StreamProcessor:
    def __init__(self):
        self.aggregators = {}
        
    def process_tick_data(self, tick: Dict) -> Dict:
        symbol = tick['symbol']
        price = tick['price']
        volume = tick['volume']
        
        if symbol not in self.aggregators:
            self.aggregators[symbol] = {
                'high': price,
                'low': price,
                'volume': 0,
                'trades': 0
            }
        
        agg = self.aggregators[symbol]
        agg['high'] = max(agg['high'], price)
        agg['low'] = min(agg['low'], price)
        agg['volume'] += volume
        agg['trades'] += 1
        
        return agg
    
    def calculate_vwap(self, symbol: str, ticks: List[Dict]) -> float:
        total_value = sum(tick['price'] * tick['volume'] for tick in ticks)
        total_volume = sum(tick['volume'] for tick in ticks)
        
        if total_volume == 0:
            return 0.0
        
        return total_value / total_volume
```

### Time-Series Database Integration

```python
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

class TimeSeriesStorage:
    def __init__(self, url: str, token: str, org: str, bucket: str):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        self.bucket = bucket
        self.org = org
        
    def write_market_data(
        self,
        symbol: str,
        price: float,
        volume: int,
        timestamp: Optional[datetime] = None
    ):
        point = Point("market_data") \
            .tag("symbol", symbol) \
            .field("price", price) \
            .field("volume", volume) \
            .time(timestamp or datetime.now(), WritePrecision.NS)
        
        self.write_api.write(bucket=self.bucket, org=self.org, record=point)
    
    def query_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict]:
        query = f'''
        from(bucket: "{self.bucket}")
        |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
        |> filter(fn: (r) => r["_measurement"] == "market_data")
        |> filter(fn: (r) => r["symbol"] == "{symbol}")
        '''
        
        result = self.query_api.query(org=self.org, query=query)
        
        data = []
        for table in result:
            for record in table.records:
                data.append({
                    'time': record.get_time(),
                    'field': record.get_field(),
                    'value': record.get_value()
                })
        
        return data
```

## Low-Latency Infrastructure

### Memory-Mapped Files for IPC

```python
import mmap
import struct
import os

class SharedMemoryMarketData:
    def __init__(self, filename: str, size: int = 1024 * 1024):
        self.filename = filename
        self.size = size
        
        if not os.path.exists(filename):
            with open(filename, 'wb') as f:
                f.write(b'\x00' * size)
        
        self.file = open(filename, 'r+b')
        self.mmap = mmap.mmap(self.file.fileno(), size)
        
    def write_price(self, offset: int, symbol: str, price: float, volume: int):
        symbol_bytes = symbol.encode('utf-8').ljust(8, b'\x00')
        
        data = struct.pack('8sdi', symbol_bytes, price, volume)
        
        self.mmap.seek(offset)
        self.mmap.write(data)
        
    def read_price(self, offset: int) -> Tuple[str, float, int]:
        self.mmap.seek(offset)
        
        data = self.mmap.read(struct.calcsize('8sdi'))
        
        symbol_bytes, price, volume = struct.unpack('8sdi', data)
        symbol = symbol_bytes.decode('utf-8').strip('\x00')
        
        return symbol, price, volume
    
    def close(self):
        self.mmap.close()
        self.file.close()
```

### Lock-Free Queue Implementation

```python
import threading
from typing import Optional, TypeVar

T = TypeVar('T')

class LockFreeQueue:
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        
    def enqueue(self, item: T) -> bool:
        next_tail = (self.tail + 1) % self.capacity
        
        if next_tail == self.head:
            return False
        
        self.buffer[self.tail] = item
        self.tail = next_tail
        
        return True
    
    def dequeue(self) -> Optional[T]:
        if self.head == self.tail:
            return None
        
        item = self.buffer[self.head]
        self.head = (self.head + 1) % self.capacity
        
        return item
    
    def is_empty(self) -> bool:
        return self.head == self.tail
    
    def size(self) -> int:
        if self.tail >= self.head:
            return self.tail - self.head
        else:
            return self.capacity - self.head + self.tail
```

## Order Management Systems

### Order Lifecycle Manager

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
import uuid

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    symbol: str
    side: str
    quantity: int
    order_type: str
    price: Optional[float] = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    fills: List[Dict] = field(default_factory=list)
    
    def add_fill(self, quantity: int, price: float):
        total_value = self.avg_fill_price * self.filled_quantity + price * quantity
        self.filled_quantity += quantity
        self.avg_fill_price = total_value / self.filled_quantity
        
        self.fills.append({
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now()
        })
        
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED


class OrderManagementSystem:
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, int] = {}
        
    def submit_order(self, order: Order) -> str:
        order.status = OrderStatus.SUBMITTED
        self.orders[order.order_id] = order
        
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
        
        order.status = OrderStatus.CANCELLED
        
        return True
    
    def process_fill(self, order_id: str, quantity: int, price: float):
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        order.add_fill(quantity, price)
        
        direction = 1 if order.side == 'buy' else -1
        self.positions[order.symbol] = self.positions.get(order.symbol, 0) + direction * quantity
    
    def get_position(self, symbol: str) -> int:
        return self.positions.get(symbol, 0)
    
    def get_open_orders(self) -> List[Order]:
        return [
            order for order in self.orders.values()
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
        ]
```

### FIX Protocol Implementation

```python
import simplefix

class FIXEngine:
    def __init__(self, sender_comp_id: str, target_comp_id: str):
        self.sender_comp_id = sender_comp_id
        self.target_comp_id = target_comp_id
        self.msg_seq_num = 1
        
    def create_new_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: Optional[float] = None,
        order_type: str = '2'
    ) -> simplefix.FixMessage:
        msg = simplefix.FixMessage()
        
        msg.append_pair(8, 'FIX.4.4')
        msg.append_pair(35, 'D')
        msg.append_pair(49, self.sender_comp_id)
        msg.append_pair(56, self.target_comp_id)
        msg.append_pair(34, self.msg_seq_num)
        msg.append_pair(52, datetime.now().strftime('%Y%m%d-%H:%M:%S'))
        
        msg.append_pair(11, str(uuid.uuid4()))
        msg.append_pair(21, '1')
        msg.append_pair(55, symbol)
        msg.append_pair(54, '1' if side.lower() == 'buy' else '2')
        msg.append_pair(60, datetime.now().strftime('%Y%m%d-%H:%M:%S'))
        msg.append_pair(38, quantity)
        msg.append_pair(40, order_type)
        
        if price and order_type == '2':
            msg.append_pair(44, price)
        
        self.msg_seq_num += 1
        
        return msg
    
    def parse_execution_report(self, fix_message: str) -> Dict:
        msg = simplefix.FixMessage()
        msg.append_string(fix_message)
        
        report = {
            'order_id': msg.get(11),
            'symbol': msg.get(55),
            'side': 'buy' if msg.get(54) == '1' else 'sell',
            'order_status': msg.get(39),
            'filled_qty': int(msg.get(14) or 0),
            'avg_price': float(msg.get(6) or 0)
        }
        
        return report
```

## Monitoring and Alerting

### Performance Monitoring

```python
import time
from collections import deque
from typing import Dict

class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        self.metrics = {
            'latency': deque(maxlen=window_size),
            'throughput': deque(maxlen=window_size),
            'errors': 0
        }
        
        self.start_times: Dict[str, float] = {}
        
    def start_timer(self, operation_id: str):
        self.start_times[operation_id] = time.perf_counter()
        
    def end_timer(self, operation_id: str):
        if operation_id in self.start_times:
            latency = (time.perf_counter() - self.start_times[operation_id]) * 1000
            
            self.metrics['latency'].append(latency)
            
            del self.start_times[operation_id]
            
    def record_throughput(self, count: int):
        self.metrics['throughput'].append(count)
        
    def record_error(self):
        self.metrics['errors'] += 1
        
    def get_statistics(self) -> Dict[str, float]:
        import numpy as np
        
        latency_arr = np.array(self.metrics['latency'])
        throughput_arr = np.array(self.metrics['throughput'])
        
        return {
            'avg_latency_ms': np.mean(latency_arr) if len(latency_arr) > 0 else 0,
            'p95_latency_ms': np.percentile(latency_arr, 95) if len(latency_arr) > 0 else 0,
            'p99_latency_ms': np.percentile(latency_arr, 99) if len(latency_arr) > 0 else 0,
            'avg_throughput': np.mean(throughput_arr) if len(throughput_arr) > 0 else 0,
            'total_errors': self.metrics['errors']
        }


class AlertManager:
    def __init__(self):
        self.alert_rules = []
        self.active_alerts = []
        
    def add_rule(self, name: str, condition: Callable[[Dict], bool], severity: str):
        self.alert_rules.append({
            'name': name,
            'condition': condition,
            'severity': severity
        })
        
    def check_metrics(self, metrics: Dict):
        for rule in self.alert_rules:
            if rule['condition'](metrics):
                alert = {
                    'name': rule['name'],
                    'severity': rule['severity'],
                    'timestamp': datetime.now(),
                    'metrics': metrics
                }
                
                self.active_alerts.append(alert)
                self._send_alert(alert)
    
    def _send_alert(self, alert: Dict):
        print(f"ALERT [{alert['severity']}]: {alert['name']}")
        print(f"Time: {alert['timestamp']}")
        print(f"Metrics: {alert['metrics']}")
```

## Testing and Deployment

### Integration Testing Framework

```python
import pytest
import asyncio

class TradingSystemTest:
    def __init__(self):
        self.oms = OrderManagementSystem()
        self.event_bus = EventBus()
        
    async def test_order_lifecycle(self):
        order = Order(
            symbol='AAPL',
            side='buy',
            quantity=100,
            order_type='limit',
            price=150.0
        )
        
        order_id = self.oms.submit_order(order)
        
        assert order_id in self.oms.orders
        assert self.oms.orders[order_id].status == OrderStatus.SUBMITTED
        
        self.oms.process_fill(order_id, 50, 150.0)
        assert self.oms.orders[order_id].status == OrderStatus.PARTIALLY_FILLED
        
        self.oms.process_fill(order_id, 50, 150.0)
        assert self.oms.orders[order_id].status == OrderStatus.FILLED
        
    async def test_position_tracking(self):
        order1 = Order(symbol='AAPL', side='buy', quantity=100, order_type='market')
        order_id1 = self.oms.submit_order(order1)
        self.oms.process_fill(order_id1, 100, 150.0)
        
        assert self.oms.get_position('AAPL') == 100
        
        order2 = Order(symbol='AAPL', side='sell', quantity=50, order_type='market')
        order_id2 = self.oms.submit_order(order2)
        self.oms.process_fill(order_id2, 50, 151.0)
        
        assert self.oms.get_position('AAPL') == 50
```

### Deployment Configuration

```python
from dataclasses import dataclass

@dataclass
class ProductionConfig:
    environment: str = "production"
    
    kafka_brokers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    redis_host: str = "localhost"
    redis_port: int = 6379
    
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = "your-token"
    influxdb_org: str = "trading-org"
    influxdb_bucket: str = "market-data"
    
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    max_position_size: int = 1000
    max_order_value: float = 100000.0
    
    log_level: str = "INFO"
    
    def validate(self):
        assert self.environment in ["production", "staging", "development"]
        assert len(self.kafka_brokers) > 0
        assert self.max_position_size > 0
        assert self.max_order_value > 0
```

## PhD-Level Research Topics

### Byzantine Fault Tolerance

```python
class ByzantineFaultTolerantConsensus:
    def __init__(self, num_nodes: int, fault_tolerance: int):
        self.num_nodes = num_nodes
        self.fault_tolerance = fault_tolerance
        self.required_agreement = (2 * fault_tolerance) + 1
        
    def propose_value(self, value: Any, node_id: int) -> bool:
        votes = {value: 1}
        
        return votes.get(value, 0) >= self.required_agreement
    
    def verify_state_consistency(
        self,
        states: List[Dict]
    ) -> bool:
        if len(states) < self.required_agreement:
            return False
        
        state_counts = {}
        for state in states:
            state_hash = hash(frozenset(state.items()))
            state_counts[state_hash] = state_counts.get(state_hash, 0) + 1
        
        max_agreement = max(state_counts.values())
        
        return max_agreement >= self.required_agreement
```

## Implementation

### Complete Production System

```python
class ProductionTradingSystem:
    def __init__(self, config: ProductionConfig):
        self.config = config
        config.validate()
        
        self.api_service = TradingAPIService(config.api_host, config.api_port)
        self.event_bus = EventBus()
        self.oms = OrderManagementSystem()
        self.performance_monitor = PerformanceMonitor()
        self.alert_manager = AlertManager()
        
        self._setup_monitoring()
        
    def _setup_monitoring(self):
        self.alert_manager.add_rule(
            "high_latency",
            lambda m: m.get('avg_latency_ms', 0) > 100,
            "warning"
        )
        
        self.alert_manager.add_rule(
            "position_limit",
            lambda m: abs(m.get('position', 0)) > self.config.max_position_size,
            "critical"
        )
        
    async def start(self):
        await asyncio.gather(
            self.event_bus.process_events(),
            self._monitor_loop()
        )
    
    async def _monitor_loop(self):
        while True:
            stats = self.performance_monitor.get_statistics()
            
            self.alert_manager.check_metrics(stats)
            
            await asyncio.sleep(60)
```
