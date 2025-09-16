# pip install web3 pandas eth-defi aiohttp asyncio
import os, json, tempfile, time
from typing import Union, Optional, Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import pandas as pd
from web3 import Web3
from web3._utils.events import get_event_data
from eth_defi.provider.multi_provider import create_multi_provider_web3

# ---------------- Config ----------------
JSON_RPC_LINE = " ".join([
    "https://eth.llamarpc.com/sk_llama_252714c1e64c9873e3b21ff94d7f1a3f",
    "https://mainnet.infura.io/v3/5f38fb376e0548c8a828112252a6a588",
    "https://snowy-broken-spring.quiknode.pro/e1c35cbb709b1cb095d49e42dcd4d40e6cbbfd7a",
    "https://worldchain-mainnet.g.alchemy.com/v2/eq9r2pPrnkczHi1MuJmXPMAc3nn3kc4F",
    "https://eth.rpc.grove.city/v1/887ffda2",
    "https://lb.nodies.app/v1/c6a2e72646e34fc78d95513a52c4aca6",
    "https://eth.drpc.org",
    "https://eth.meowrpc.com",
    "https://ethereum.publicnode.com"
])

POOL_ADDR = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"  # USDC/WETH 0.05%

# Provide either UNIX seconds or ISO strings (UTC assumed unless offset specified)
START_TS: Union[int, str] = "2023-01-01T00:00:00Z"
END_TS:   Union[int, str] = "2023-12-31T23:59:59Z"

# OPTIMIZED: Adaptive chunks, parallel processing
CHUNK_SIZE_BLOCKS = 2000  # Reduced to avoid provider limits
PARALLEL_WORKERS = 8  # Parallel chunk processing
BATCH_RECEIPT_SIZE = 100  # Batch receipt fetching

# Checkpointing
CHECKPOINT_PATH = "univ3_checkpoint.json"
OUT_CSV = "univ3_pool_events_with_running_state.csv"

# Option to skip gas data entirely (HUGE speedup)
SKIP_GAS_DATA = True  # Set to False if you really need gas data

# ---------------- Setup ----------------
w3 = create_multi_provider_web3(JSON_RPC_LINE, request_kwargs={"timeout": 60.0})
POOL = Web3.to_checksum_address(POOL_ADDR)

Q96 = 1 << 96

def to_unix(ts: Union[int, str]) -> int:
    if isinstance(ts, int):
        return ts
    return int(pd.to_datetime(ts, utc=True).value // 10**9)

START_TS = to_unix(START_TS)
END_TS   = to_unix(END_TS)
if END_TS < START_TS:
    raise ValueError("END_TS must be >= START_TS")

# ---------------- ABIs (unchanged) ----------------
SWAP_EVENT_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": True,  "internalType":"address","name":"sender","type":"address"},
        {"indexed": True,  "internalType":"address","name":"recipient","type":"address"},
        {"indexed": False, "internalType":"int256", "name":"amount0","type":"int256"},
        {"indexed": False, "internalType":"int256", "name":"amount1","type":"int256"},
        {"indexed": False, "internalType":"uint160","name":"sqrtPriceX96","type":"uint160"},
        {"indexed": False, "internalType":"uint128","name":"liquidity","type":"uint128"},
        {"indexed": False, "internalType":"int24",  "name":"tick","type":"int24"}
    ],
    "name": "Swap", "type": "event"
}
MINT_EVENT_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": False, "internalType":"address","name":"sender","type":"address"},
        {"indexed": True,  "internalType":"address","name":"owner","type":"address"},
        {"indexed": True,  "internalType":"int24",  "name":"tickLower","type":"int24"},
        {"indexed": True,  "internalType":"int24",  "name":"tickUpper","type":"int24"},
        {"indexed": False, "internalType":"uint128","name":"amount","type":"uint128"},
        {"indexed": False, "internalType":"uint256","name":"amount0","type":"uint256"},
        {"indexed": False, "internalType":"uint256","name":"amount1","type":"uint256"}
    ],
    "name": "Mint", "type": "event"
}
BURN_EVENT_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": True,  "internalType":"address","name":"owner","type":"address"},
        {"indexed": True,  "internalType":"int24",  "name":"tickLower","type":"int24"},
        {"indexed": True,  "internalType":"int24",  "name":"tickUpper","type":"int24"},
        {"indexed": False, "internalType":"uint128","name":"amount","type":"uint128"},
        {"indexed": False, "internalType":"uint256","name":"amount0","type":"uint256"},
        {"indexed": False, "internalType":"uint256","name":"amount1","type":"uint256"}
    ],
    "name": "Burn", "type": "event"
}
POOL_STATE_ABI = [
    {"name":"slot0","inputs":[],"outputs":[
        {"type":"uint160","name":"sqrtPriceX96"},
        {"type":"int24","name":"tick"},
        {"type":"uint16","name":"observationIndex"},
        {"type":"uint16","name":"observationCardinality"},
        {"type":"uint16","name":"observationCardinalityNext"},
        {"type":"uint8","name":"feeProtocol"},
        {"type":"bool","name":"unlocked"}],
     "stateMutability":"view","type":"function"},
    {"name":"liquidity","inputs":[],"outputs":[{"type":"uint128"}],
     "stateMutability":"view","type":"function"},
]

contract = w3.eth.contract(address=POOL, abi=[SWAP_EVENT_ABI, MINT_EVENT_ABI, BURN_EVENT_ABI])
state_c  = w3.eth.contract(address=POOL, abi=POOL_STATE_ABI)

SwapEvent = contract.events.Swap
MintEvent = contract.events.Mint
BurnEvent = contract.events.Burn

SWAP_TOPIC0 = w3.keccak(text="Swap(address,address,int256,int256,uint160,uint128,int24)").hex()
MINT_TOPIC0 = w3.keccak(text="Mint(address,address,int24,int24,uint128,uint256,uint256)").hex()
BURN_TOPIC0 = w3.keccak(text="Burn(address,int24,int24,uint128,uint256,uint256)").hex()

# ---------------- Block helpers with caching ----------------
_block_cache = {}
_ts_cache = {}

def get_block_cached(block_num: int):
    if block_num not in _block_cache:
        _block_cache[block_num] = w3.eth.get_block(block_num)
    return _block_cache[block_num]

def block_ts(bn: int) -> int:
    if bn not in _ts_cache:
        _ts_cache[bn] = get_block_cached(bn)["timestamp"]
    return _ts_cache[bn]

def block_for_timestamp(target_ts: int, mode: str = "start") -> int:
    latest = w3.eth.block_number
    lo, hi = 0, latest
    ans = None
    while lo <= hi:
        mid = (lo + hi) // 2
        ts = get_block_cached(mid)["timestamp"]
        if mode == "start":
            if ts >= target_ts:
                ans, hi = mid, mid - 1
            else:
                lo = mid + 1
        else:
            if ts <= target_ts:
                ans, lo = mid, mid + 1
            else:
                hi = mid - 1
    return ans if ans is not None else (0 if mode == "start" else 0)

START_BLOCK = block_for_timestamp(START_TS, "start")
END_BLOCK   = block_for_timestamp(END_TS,   "end")

# ---------------- OPTIMIZED: Batch receipt and transaction fetching ----------------
_rcpt_cache: Dict[str, Any] = {}
_tx_cache: Dict[str, Any] = {}  # Cache for transaction data

def batch_fetch_receipts_and_txs(tx_hashes: List[str]) -> tuple[Dict[str, Optional[Dict]], Dict[str, Optional[str]]]:
    """Fetch multiple receipts and transaction origins in parallel"""
    receipt_result = {}
    origin_result = {}
    to_fetch = [h for h in tx_hashes if h not in _rcpt_cache]
    
    if not to_fetch:
        # Everything cached
        for h in tx_hashes:
            receipt_result[h] = _rcpt_cache.get(h)
            origin_result[h] = _tx_cache.get(h, {}).get('from')
        return receipt_result, origin_result
    
    if SKIP_GAS_DATA:
        # Still fetch origins even if skipping gas
        def fetch_tx(txh):
            try:
                if txh not in _tx_cache:
                    tx = w3.eth.get_transaction(txh)
                    _tx_cache[txh] = {'from': tx['from']}
                return txh, _tx_cache[txh]['from']
            except:
                return txh, None
        
        with ThreadPoolExecutor(max_workers=min(10, len(to_fetch))) as executor:
            futures = [executor.submit(fetch_tx, txh) for txh in to_fetch]
            for future in as_completed(futures):
                txh, origin = future.result()
                origin_result[txh] = origin
        
        for h in tx_hashes:
            receipt_result[h] = _rcpt_cache.get(h)
            if h not in origin_result:
                origin_result[h] = _tx_cache.get(h, {}).get('from')
    else:
        # Fetch both receipts and transactions
        def fetch_both(txh):
            try:
                receipt = w3.eth.get_transaction_receipt(txh) if txh not in _rcpt_cache else _rcpt_cache[txh]
                tx = w3.eth.get_transaction(txh) if txh not in _tx_cache else _tx_cache[txh]
                if receipt and txh not in _rcpt_cache:
                    _rcpt_cache[txh] = receipt
                if tx and txh not in _tx_cache:
                    _tx_cache[txh] = {'from': tx['from']}
                return txh, receipt, tx['from'] if tx else None
            except:
                return txh, None, None
        
        with ThreadPoolExecutor(max_workers=min(10, len(to_fetch))) as executor:
            futures = [executor.submit(fetch_both, txh) for txh in to_fetch]
            for future in as_completed(futures):
                txh, receipt, origin = future.result()
                if receipt:
                    _rcpt_cache[txh] = receipt
                receipt_result[txh] = receipt
                origin_result[txh] = origin
        
        for h in tx_hashes:
            if h not in receipt_result:
                receipt_result[h] = _rcpt_cache.get(h)
            if h not in origin_result:
                origin_result[h] = _tx_cache.get(h, {}).get('from')
    
    return receipt_result, origin_result

# ---------------- OPTIMIZED: Faster getLogs with better retry ----------------
RETRYABLE = (
    "timeout", "503", "502", "500", "429", "rate limit", "too many",
    "limit exceeded", "gateway", "413", "entity too large", 
    "payload too large", "request entity too large", "content too big",
    "range is too large", "max is 1k blocks",
    "query returned more than 10000 results", "exceeds max results",
    "-32005", "-32603", "-32602",  # Added -32602 for the specific error
)

def get_logs_chunked_any(topics_any: list[str], start_block: int, end_block: int):
    """Optimized log fetching with aggressive adaptive chunking"""
    SOFT_LOG_LIMIT = 5000  # Reduced to be safer with providers
    
    def extract_suggested_range(error_msg: str):
        """Extract suggested range from error message if present"""
        import re
        match = re.search(r'range (\d+)-(\d+)', error_msg)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None
    
    def yield_range(a: int, b: int, retry_count: int = 0):
        # Aggressive reduction after retries
        if retry_count > 2:
            # Use very small chunks after multiple retries
            chunk_size = max((b - a) // 10, 50)
            for chunk_start in range(a, b + 1, chunk_size):
                chunk_end = min(chunk_start + chunk_size - 1, b)
                yield from yield_range(chunk_start, chunk_end, 0)
            return
        
        filt = {
            "fromBlock": a,
            "toBlock": b,
            "address": POOL,
            "topics": [topics_any],
        }
        
        try:
            logs = w3.eth.get_logs(filt)
        except Exception as e:
            error_str = str(e)
            msg = error_str.lower()
            
            # Check if we can split the range
            if b > a:
                # Check for specific "exceeds max results" error with suggested range
                if "exceeds max results" in msg:
                    suggested = extract_suggested_range(error_str)
                    if suggested:
                        # Use the suggested range
                        suggest_start, suggest_end = suggested
                        if suggest_start == a:
                            # Provider gave us a max range starting from our start
                            yield from yield_range(a, suggest_end, 0)
                            if suggest_end < b:
                                yield from yield_range(suggest_end + 1, b, 0)
                            return
                    
                    # No suggested range, split aggressively
                    # For 20k result limit, split into 8 parts
                    part_size = (b - a) // 8
                    for i in range(8):
                        start = a + i * part_size
                        end = a + (i + 1) * part_size - 1 if i < 7 else b
                        if start <= end:
                            yield from yield_range(start, end, retry_count + 1)
                    return
                
                # Check for other retryable errors
                elif any(s in msg for s in RETRYABLE):
                    # Determine split strategy based on error type
                    if "10000" in msg or "20000" in msg:
                        # Very aggressive split for result size limits
                        parts = 8
                    elif "entity too large" in msg or "payload too large" in msg:
                        # Large payload, split into 4
                        parts = 4
                    else:
                        # Default binary split
                        parts = 2
                    
                    part_size = (b - a + 1) // parts
                    for i in range(parts):
                        start = a + i * part_size
                        end = a + (i + 1) * part_size - 1 if i < parts - 1 else b
                        if start <= end:
                            yield from yield_range(start, end, retry_count + 1)
                    return
            
            # Can't split further or not a retryable error
            raise
        
        # Check if response is large and proactively split
        if len(logs) > SOFT_LOG_LIMIT and b > a:
            # Split proactively to avoid issues
            mid = (a + b) // 2
            yield from yield_range(a, mid, retry_count)
            yield from yield_range(mid + 1, b, retry_count)
            return
        
        for lg in logs:
            yield lg
    
    # Process in chunks
    cur = start_block
    while cur <= end_block:
        to_blk = min(cur + CHUNK_SIZE_BLOCKS - 1, end_block)
        try:
            yield from yield_range(cur, to_blk)
        except Exception as e:
            print(f"  ⚠️  Error in range {cur}-{to_blk}: {str(e)[:100]}")
            # Fall back to smaller chunks
            smaller_chunk = max(CHUNK_SIZE_BLOCKS // 4, 100)
            inner_cur = cur
            while inner_cur <= to_blk:
                inner_end = min(inner_cur + smaller_chunk - 1, to_blk)
                yield from yield_range(inner_cur, inner_end)
                inner_cur = inner_end + 1
        
        cur = to_blk + 1

# ---------------- Checkpoint helpers (unchanged) ----------------
def load_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def atomic_write_json(path: str, payload: Dict[str, Any]):
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".ckpt_", dir=d, text=True)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, separators=(",", ":"), sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try: os.remove(tmp)
        except Exception: pass
        raise

def save_checkpoint(path: str, data: Dict[str, Any]): 
    atomic_write_json(path, data)

def csv_append(df: pd.DataFrame, path: str):
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)

# ---------------- Virtual balances ----------------
def virt_x(L, sP): return (int(L) * Q96) // int(sP) if sP else None
def virt_y(L, sP): return (int(L) * int(sP)) // Q96 if sP else None

# ---------------- OPTIMIZED: Parallel log processing ----------------
def process_logs_parallel(logs_chunks: List[List], start_block: int, end_block: int):
    """Process multiple log chunks in parallel"""
    all_rows = []
    
    def process_chunk(logs):
        rows = []
        for log in logs:
            topic0 = log["topics"][0].hex()
            bn = log["blockNumber"]
            ts = block_ts(bn)
            if ts < START_TS or ts > END_TS:
                continue
            txh = log["transactionHash"].hex()
            
            if topic0 == SWAP_TOPIC0:
                evt = get_event_data(w3.codec, SwapEvent._get_event_abi(), log)
                a = evt["args"]
                rows.append({
                    "eventType": "Swap",
                    "blockNumber": bn, "logIndex": log["logIndex"], "timestamp": ts,
                    "transactionHash": txh, "gasUsed": None, "origin": None,
                    "sender": a["sender"], "owner": None, "recipient": a["recipient"],
                    "amount0": int(a["amount0"]), "amount1": int(a["amount1"]),
                    "sqrtPriceX96_event": int(a["sqrtPriceX96"]),
                    "tick_event": int(a["tick"]),
                    "liquidityAfter_event": int(a["liquidity"]),
                    "tickLower": None, "tickUpper": None,
                    "liquidityDelta": None
                })
            elif topic0 == MINT_TOPIC0:
                evt = get_event_data(w3.codec, MintEvent._get_event_abi(), log)
                a = evt["args"]
                rows.append({
                    "eventType": "Mint",
                    "blockNumber": bn, "logIndex": log["logIndex"], "timestamp": ts,
                    "transactionHash": txh, "gasUsed": None, "origin": None,
                    "sender": a["sender"], "owner": a["owner"], "recipient": None,
                    "amount0": int(a["amount0"]), "amount1": int(a["amount1"]),
                    "sqrtPriceX96_event": None, "tick_event": None, "liquidityAfter_event": None,
                    "tickLower": int(a["tickLower"]), "tickUpper": int(a["tickUpper"]),
                    "liquidityDelta": int(a["amount"])
                })
            elif topic0 == BURN_TOPIC0:
                evt = get_event_data(w3.codec, BurnEvent._get_event_abi(), log)
                a = evt["args"]
                rows.append({
                    "eventType": "Burn",
                    "blockNumber": bn, "logIndex": log["logIndex"], "timestamp": ts,
                    "transactionHash": txh, "gasUsed": None, "origin": None,
                    "sender": None, "owner": a["owner"], "recipient": None,
                    "amount0": int(a["amount0"]), "amount1": int(a["amount1"]),
                    "sqrtPriceX96_event": None, "tick_event": None, "liquidityAfter_event": None,
                    "tickLower": int(a["tickLower"]), "tickUpper": int(a["tickUpper"]),
                    "liquidityDelta": -int(a["amount"])
                })
        return rows
    
    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in logs_chunks]
        for future in as_completed(futures):
            all_rows.extend(future.result())
    
    return all_rows

# ---------------- OPTIMIZED: Main processing function ----------------
def process_window(from_block: int, to_block: int, 
                   cur_L: int, cur_sqrt: int, cur_tick: int) -> Dict[str, Any]:
    """Optimized window processing with parallel operations"""
    
    # Collect all logs first
    print(f"  Fetching logs for blocks {from_block}-{to_block}...")
    all_logs = list(get_logs_chunked_any([SWAP_TOPIC0, MINT_TOPIC0, BURN_TOPIC0], 
                                          from_block, to_block))
    
    if not all_logs:
        return {
            "df_chunk": pd.DataFrame(),
            "new_cur_L": cur_L, "new_cur_sqrt": cur_sqrt, "new_cur_tick": cur_tick,
            "first_event_block": None, "last_event_block": None, "n_rows": 0
        }
    
    print(f"  Processing {len(all_logs)} logs...")
    
    # Split logs into chunks for parallel processing
    chunk_size = max(len(all_logs) // PARALLEL_WORKERS, 100)
    log_chunks = [all_logs[i:i+chunk_size] for i in range(0, len(all_logs), chunk_size)]
    
    # Process in parallel
    rows = process_logs_parallel(log_chunks, from_block, to_block)
    
    if not rows:
        return {
            "df_chunk": pd.DataFrame(),
            "new_cur_L": cur_L, "new_cur_sqrt": cur_sqrt, "new_cur_tick": cur_tick,
            "first_event_block": None, "last_event_block": None, "n_rows": 0
        }
    
    df = pd.DataFrame(rows).sort_values(["blockNumber","logIndex"]).reset_index(drop=True)
    
    # Batch fetch receipts and origins if needed
    if not SKIP_GAS_DATA:
        print(f"  Fetching receipts and origins for {df['transactionHash'].nunique()} transactions...")
        tx_list = df["transactionHash"].unique().tolist()
        
        # Process in batches
        gas_map = {}
        origin_map = {}
        for i in range(0, len(tx_list), BATCH_RECEIPT_SIZE):
            batch = tx_list[i:i+BATCH_RECEIPT_SIZE]
            receipts, origins = batch_fetch_receipts_and_txs(batch)
            for txh in batch:
                rc = receipts.get(txh)
                gas_map[txh] = rc["gasUsed"] if rc and "gasUsed" in rc else None
                origin_map[txh] = origins.get(txh)
        
        df["gasUsed"] = df["transactionHash"].map(gas_map)
        df["origin"] = df["transactionHash"].map(origin_map)
    else:
        # Still fetch origins even when skipping gas
        print(f"  Fetching origins for {df['transactionHash'].nunique()} transactions...")
        tx_list = df["transactionHash"].unique().tolist()
        
        origin_map = {}
        for i in range(0, len(tx_list), BATCH_RECEIPT_SIZE):
            batch = tx_list[i:i+BATCH_RECEIPT_SIZE]
            _, origins = batch_fetch_receipts_and_txs(batch)
            for txh in batch:
                origin_map[txh] = origins.get(txh)
        
        df["origin"] = df["transactionHash"].map(origin_map)
    
    # Running state calculation (unchanged but optimized with vectorization where possible)
    print(f"  Computing running state...")
    pre_L, pre_sqrt, pre_tick = [], [], []
    post_L, post_sqrt, post_tick = [], [], []
    x_before, y_before, x_after, y_after = [], [], [], []
    affects_active, delta_applied = [], []
    
    curL, curSP, curTk = int(cur_L), int(cur_sqrt), int(cur_tick)
    
    for _, row in df.iterrows():
        etype = row["eventType"]
        
        pre_L.append(curL); pre_sqrt.append(curSP); pre_tick.append(curTk)
        x_before.append(virt_x(curL, curSP))
        y_before.append(virt_y(curL, curSP))
        
        if etype in ("Mint", "Burn"):
            hit = (row["tickLower"] <= curTk) and (curTk < row["tickUpper"])
            affects_active.append(bool(hit))
            dL = int(row["liquidityDelta"]) if hit else 0
            delta_applied.append(dL)
            curL = curL + dL
            post_L.append(curL); post_sqrt.append(curSP); post_tick.append(curTk)
        elif etype == "Swap":
            affects_active.append(None)
            delta_applied.append(None)
            curL = int(row["liquidityAfter_event"])
            curSP = int(row["sqrtPriceX96_event"])
            curTk = int(row["tick_event"])
            post_L.append(curL); post_sqrt.append(curSP); post_tick.append(curTk)
        
        x_after.append(virt_x(post_L[-1], post_sqrt[-1]))
        y_after.append(virt_y(post_L[-1], post_sqrt[-1]))
    
    # Batch assign all columns at once
    df["L_before"] = pre_L
    df["sqrt_before"] = pre_sqrt
    df["tick_before"] = pre_tick
    df["x_before"] = x_before
    df["y_before"] = y_before
    df["L_after"] = post_L
    df["sqrt_after"] = post_sqrt
    df["tick_after"] = post_tick
    df["x_after"] = x_after
    df["y_after"] = y_after
    df["affectsActive"] = affects_active
    df["deltaL_applied"] = delta_applied
    
    return {
        "df_chunk": df,
        "new_cur_L": curL, "new_cur_sqrt": curSP, "new_cur_tick": curTk,
        "first_event_block": int(df.iloc[0]["blockNumber"]),
        "last_event_block": int(df.iloc[-1]["blockNumber"]),
        "n_rows": len(df),
    }

# ---------------- Main execution ----------------
print(f"Starting optimized data fetch for pool {POOL_ADDR}")
print(f"Date range: {pd.to_datetime(START_TS, unit='s')} to {pd.to_datetime(END_TS, unit='s')}")
print(f"Block range: {START_BLOCK} to {END_BLOCK} (~{END_BLOCK - START_BLOCK:,} blocks)")
print(f"Settings: CHUNK_SIZE={CHUNK_SIZE_BLOCKS}, WORKERS={PARALLEL_WORKERS}, SKIP_GAS={SKIP_GAS_DATA}")

ckpt = load_checkpoint(CHECKPOINT_PATH)
fresh = True

if ckpt:
    ok = (
        ckpt.get("pool") == POOL_ADDR
        and ckpt.get("start_ts") == START_TS
        and ckpt.get("end_ts") == END_TS
        and ckpt.get("start_block") == START_BLOCK
        and ckpt.get("end_block") == END_BLOCK
    )
    if ok:
        from_block = int(ckpt["next_from_block"])
        cur_L = int(ckpt["cur_L"])
        cur_sqrt = int(ckpt["cur_sqrt"])
        cur_tick = int(ckpt["cur_tick"])
        fresh = False
        print(f"Resuming from block {from_block} (state: L={cur_L}, sP={cur_sqrt}, tick={cur_tick})")
    else:
        print("Checkpoint params differ. Starting fresh.")
        ckpt = None

if fresh:
    from_block = START_BLOCK
    prev_block = max(from_block - 1, 0)
    s0 = state_c.functions.slot0().call(block_identifier=prev_block)
    L0 = int(state_c.functions.liquidity().call(block_identifier=prev_block))
    cur_sqrt = int(s0[0]); cur_tick = int(s0[1]); cur_L = int(L0)
    
    if os.path.exists(OUT_CSV):
        print(f"Removing previous output {OUT_CSV}")
        os.remove(OUT_CSV)
    
    ckpt = {
        "pool": POOL_ADDR,
        "start_ts": START_TS,
        "end_ts": END_TS,
        "start_block": START_BLOCK,
        "end_block": END_BLOCK,
        "next_from_block": from_block,
        "cur_L": cur_L,
        "cur_sqrt": cur_sqrt,
        "cur_tick": cur_tick,
        "events_written": 0,
        "last_event_block": None,
        "out_csv": OUT_CSV,
    }
    save_checkpoint(CHECKPOINT_PATH, ckpt)
    print(f"Initial state at block {prev_block}: L={cur_L}, sP={cur_sqrt}, tick={cur_tick}")

# Progress tracking
import datetime
start_time = time.time()
initial_block = from_block
total_blocks = END_BLOCK - START_BLOCK + 1

while from_block <= END_BLOCK:
    to_block = min(from_block + CHUNK_SIZE_BLOCKS - 1, END_BLOCK)
    
    # Progress reporting
    blocks_done = from_block - initial_block
    pct_done = (blocks_done / total_blocks) * 100 if total_blocks > 0 else 0
    elapsed = time.time() - start_time
    if blocks_done > 0:
        rate = blocks_done / elapsed
        eta = (total_blocks - blocks_done) / rate if rate > 0 else 0
        eta_str = str(datetime.timedelta(seconds=int(eta)))
        print(f"\n[{pct_done:.1f}%] Processing blocks [{from_block:,}, {to_block:,}] "
              f"(Rate: {rate:.0f} blocks/s, ETA: {eta_str})")
    else:
        print(f"\nProcessing blocks [{from_block:,}, {to_block:,}]...")
    
    result = process_window(from_block, to_block, cur_L, cur_sqrt, cur_tick)
    df_chunk = result["df_chunk"]
    n_rows = result["n_rows"]
    
    cur_L = result["new_cur_L"]
    cur_sqrt = result["new_cur_sqrt"]
    cur_tick = result["new_cur_tick"]
    
    if n_rows > 0:
        csv_append(df_chunk, OUT_CSV)
        
        ckpt.update({
            "cur_L": cur_L,
            "cur_sqrt": cur_sqrt,
            "cur_tick": cur_tick,
            "events_written": int(ckpt.get("events_written", 0)) + n_rows,
            "last_event_block": result["last_event_block"],
            "next_from_block": to_block + 1,
        })
        save_checkpoint(CHECKPOINT_PATH, ckpt)
        print(f"  ✓ Wrote {n_rows} rows | Last event block: {result['last_event_block']}")
    else:
        ckpt.update({
            "cur_L": cur_L,
            "cur_sqrt": cur_sqrt,
            "cur_tick": cur_tick,
            "next_from_block": to_block + 1,
        })
        save_checkpoint(CHECKPOINT_PATH, ckpt)
        print("  ✓ No events in window")
    
    from_block = to_block + 1

# Final summary
elapsed_total = time.time() - start_time
print(f"\n{'='*60}")
print(f"✅ COMPLETED!")
print(f"Time range: {pd.to_datetime(START_TS, unit='s')} to {pd.to_datetime(END_TS, unit='s')}")
print(f"Total events written: {ckpt.get('events_written', 0):,}")
print(f"Output file: {OUT_CSV}")
print(f"Total time: {str(datetime.timedelta(seconds=int(elapsed_total)))}")
print(f"Average rate: {total_blocks/elapsed_total:.0f} blocks/second")
print(f"{'='*60}")