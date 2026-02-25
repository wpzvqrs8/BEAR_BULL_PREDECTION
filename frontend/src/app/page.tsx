"use client";

import React, { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import { Activity, TrendingUp, TrendingDown, Play, Pause, BarChart2, LineChart as LineChartIcon, CheckCircle, XCircle, Clock, ChevronDown } from 'lucide-react';
import TradingPanel from './components/TradingPanel';

const LightweightChart = dynamic(() => import('./components/ChartComponent'), {
  ssr: false,
  loading: () => <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-sm">Initializing Chart Engine...</div>
});

const TIMEFRAMES = [
  { label: '1 Min', value: '1m' },
  { label: '5 Min', value: '5m' },
  { label: '15 Min', value: '15m' },
  { label: '30 Min', value: '30m' },
  { label: '1 Hour', value: '1h' },
  { label: '1 Day', value: '1d' }
];

const ASSETS = [
  { label: 'BTC-USD', value: 'BTC-USD', icon: '₿', color: 'text-yellow-400' },
  { label: 'ETH-USD', value: 'ETH-USD', icon: 'Ξ', color: 'text-purple-400' },
  { label: 'GOLD', value: 'GOLD', icon: '◈', color: 'text-amber-500' },
];

type PredictionRecord = {
  id: number;
  predicted: 'bull' | 'bear';
  actual: 'bull' | 'bear' | null;
  bull_prob: number;
  bear_prob: number;
  price: number;
  correct: boolean | null;
  reason?: string;
};

export default function Home() {
  const [liveData, setLiveData] = useState<any>(null);
  const [currentPrediction, setCurrentPrediction] = useState<any>(null);
  const [history, setHistory] = useState<PredictionRecord[]>([]);
  const [isPlaying, setIsPlaying] = useState(true);
  const [activeTimeframe, setActiveTimeframe] = useState('1m');
  const [activeSymbol, setActiveSymbol] = useState('BTC-USD');
  const [chartType, setChartType] = useState<'candle' | 'line'>('candle');
  const [countdown, setCountdown] = useState<Record<number, number>>({});
  const [marketClosed, setMarketClosed] = useState(false);
  const [assetDropdown, setAssetDropdown] = useState(false);
  const [currentPrice, setCurrentPrice] = useState(0);
  const [isClient, setIsClient] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const isPlayingRef = useRef(true);
  const pendingIdRef = useRef<number | null>(null);

  useEffect(() => { setIsClient(true); }, []);
  useEffect(() => { isPlayingRef.current = isPlaying; }, [isPlaying]);

  const resolved = history.filter(h => h.correct !== null);
  const correct = resolved.filter(h => h.correct === true).length;
  const accuracy = resolved.length > 0 ? ((correct / resolved.length) * 100).toFixed(1) : '—';

  const activeAsset = ASSETS.find(a => a.value === activeSymbol) ?? ASSETS[0];

  // ── WebSocket effect — reconnects on symbol OR timeframe change ──────────
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/stream');
    wsRef.current = ws;
    setMarketClosed(false);
    setHistory([]);
    setCurrentPrediction(null);

    ws.onopen = () => {
      ws.send(JSON.stringify({ symbol: activeSymbol, interval: activeTimeframe }));
    };

    ws.onmessage = (event) => {
      if (!isPlayingRef.current) return;
      const data = JSON.parse(event.data);

      if (data.type === 'market_closed') {
        setMarketClosed(true);
        setCurrentPrediction(null);

      } else if (data.type === 'history') {
        setMarketClosed(false);
        setLiveData(data);

      } else if (data.type === 'prediction') {
        setMarketClosed(false);
        pendingIdRef.current = data.candle_id;
        setCurrentPrediction(data);
        if (data.current_price) setCurrentPrice(data.current_price);
        setHistory(prev => [{
          id: data.candle_id,
          predicted: data.predicted_direction,
          actual: null,
          bull_prob: data.bull_probability,
          bear_prob: data.bear_probability,
          price: data.current_price,
          correct: null,
        }, ...prev]);
        if (data.resolve_in_seconds) {
          setCountdown(prev => ({ ...prev, [data.candle_id]: data.resolve_in_seconds }));
        }

      } else if (data.type === 'prediction_update') {
        setCurrentPrediction(data);
        if (data.current_price) setCurrentPrice(data.current_price);
        if (pendingIdRef.current !== null && data.remaining_seconds !== undefined) {
          setCountdown(prev => ({ ...prev, [pendingIdRef.current!]: Math.ceil(data.remaining_seconds) }));
        }

      } else if (data.type === 'candle') {
        setLiveData(data);
        if (data.price) setCurrentPrice(data.price);
        setCurrentPrediction(null);
        pendingIdRef.current = null;
        const finalDir = data.final_predicted || null;
        setHistory(prev => prev.map(r => {
          if (r.id !== data.candle_id) return r;
          const scored = finalDir
            ? finalDir === data.actual_direction
            : r.predicted === data.actual_direction;
          return {
            ...r,
            predicted: (finalDir as any) || r.predicted,
            actual: data.actual_direction,
            correct: scored,
            reason: !scored ? data.reason : undefined,
          };
        }));

      } else if (data.type === 'chart_candle') {
        setLiveData(data);
        if (data.price) setCurrentPrice(data.price);
      }
    };

    return () => ws.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSymbol, activeTimeframe]);

  const handleTimeframeChange = (tf: string) => {
    setActiveTimeframe(tf);
    setCurrentPrediction(null);
  };

  const handleAssetChange = (sym: string) => {
    setActiveSymbol(sym);
    setAssetDropdown(false);
    setCurrentPrediction(null);
    setHistory([]);
  };

  const fmtCountdown = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m}:${String(s).padStart(2, '0')}`;
  };

  if (!isClient) return null;

  return (
    <main className="min-h-screen p-6 grid grid-cols-12 gap-6 bg-[#09090b]">

      {/* ── Left Sidebar ── */}
      <div className="col-span-3 space-y-4">

        {/* Asset + Controls */}
        <div className="glass-panel p-5">
          <div className="flex items-center gap-2 mb-5">
            <Activity className="text-blue-500" size={20} />
            <h1 className="text-lg font-bold tracking-tight">AI Quant Engine</h1>
          </div>

          <div className="space-y-4">
            {/* ── Asset Selector ── */}
            <div className="relative">
              <label className="text-xs text-gray-500 uppercase tracking-wider block mb-1.5">Target Asset</label>
              <button
                onClick={() => setAssetDropdown(!assetDropdown)}
                className="w-full flex items-center justify-between bg-[#27272a] hover:bg-[#3f3f46] border border-[#3f3f46] rounded-lg px-3 py-2.5 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <span className={`text-xl font-bold ${activeAsset.color}`}>{activeAsset.icon}</span>
                  <span className="font-semibold text-sm">{activeAsset.label}</span>
                </div>
                <ChevronDown size={14} className={`text-gray-400 transition-transform ${assetDropdown ? 'rotate-180' : ''}`} />
              </button>

              {assetDropdown && (
                <div className="absolute z-50 top-full mt-1 w-full bg-[#18181b] border border-[#3f3f46] rounded-lg shadow-2xl overflow-hidden">
                  {ASSETS.map(asset => (
                    <button
                      key={asset.value}
                      onClick={() => handleAssetChange(asset.value)}
                      className={`w-full flex items-center gap-3 px-3 py-2.5 hover:bg-[#27272a] transition-colors text-left ${activeSymbol === asset.value ? 'bg-blue-600/15' : ''}`}
                    >
                      <span className={`text-lg font-bold ${asset.color}`}>{asset.icon}</span>
                      <div>
                        <div className="text-sm font-semibold">{asset.label}</div>
                        <div className="text-[10px] text-gray-500">
                          {asset.value === 'BTC-USD' ? 'Bitcoin · Binance Live' :
                            asset.value === 'ETH-USD' ? 'Ethereum · Binance Live' :
                              'Gold Futures · COMEX / yfinance'}
                        </div>
                      </div>
                      {activeSymbol === asset.value && <CheckCircle size={13} className="ml-auto text-blue-400" />}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="pt-3 border-t border-[#27272a]">
              <label className="text-xs text-gray-500 uppercase tracking-wider mb-2 block">Timeframe</label>
              <div className="flex flex-wrap gap-1.5">
                {TIMEFRAMES.map(tf => (
                  <button
                    key={tf.value}
                    onClick={() => handleTimeframeChange(tf.value)}
                    className={`px-2.5 py-1 text-xs rounded-md transition-colors ${activeTimeframe === tf.value ? 'bg-blue-600 text-white' : 'bg-[#27272a] hover:bg-[#3f3f46] text-gray-300'}`}
                  >
                    {tf.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Live Prediction Meter */}
        <div className="glass-panel p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Next Candle Prediction</h2>
            {currentPrediction ? (
              <span className="flex items-center gap-1 text-[10px] text-yellow-400 animate-pulse">
                <Clock size={10} /> Predicting...
              </span>
            ) : marketClosed ? (
              <span className="text-[10px] text-orange-400">Market Closed</span>
            ) : (
              <span className="text-[10px] text-gray-600">Awaiting cycle</span>
            )}
          </div>

          <div className="space-y-4">
            <div>
              <div className="flex justify-between mb-1.5">
                <span className={`text-sm flex items-center gap-1 ${currentPrediction?.predicted_direction === 'bull' ? 'text-green-300 font-semibold' : 'text-green-500'}`}>
                  <TrendingUp size={14} /> Bull
                  {currentPrediction?.predicted_direction === 'bull' && <span className="ml-1 text-[10px] bg-green-500/20 text-green-400 px-1.5 py-0.5 rounded">PREDICTED</span>}
                </span>
                <span className="font-mono text-sm">{((currentPrediction?.bull_probability ?? 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-green-600 to-green-400 transition-all duration-700" style={{ width: `${(currentPrediction?.bull_probability ?? 0) * 100}%` }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between mb-1.5">
                <span className={`text-sm flex items-center gap-1 ${currentPrediction?.predicted_direction === 'bear' ? 'text-red-300 font-semibold' : 'text-red-500'}`}>
                  <TrendingDown size={14} /> Bear
                  {currentPrediction?.predicted_direction === 'bear' && <span className="ml-1 text-[10px] bg-red-500/20 text-red-400 px-1.5 py-0.5 rounded">PREDICTED</span>}
                </span>
                <span className="font-mono text-sm">{((currentPrediction?.bear_probability ?? 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 w-full bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-gradient-to-r from-red-600 to-red-400 transition-all duration-700" style={{ width: `${(currentPrediction?.bear_probability ?? 0) * 100}%` }} />
              </div>
            </div>

            <div className="pt-3 border-t border-[#27272a]">
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs text-gray-500">Model Confidence</span>
                <span className="font-mono text-lg font-bold text-blue-400">
                  {currentPrediction ? `${(currentPrediction.confidence_score * 100).toFixed(1)}%` : '—'}
                </span>
              </div>
              {currentPrediction?.model && (
                <span className="text-[9px] font-mono text-gray-600 bg-[#27272a] px-1.5 py-0.5 rounded">
                  {currentPrediction.model}
                </span>
              )}
            </div>

            {currentPrediction?.detected_patterns?.length > 0 && (
              <div className="pt-3 border-t border-[#27272a]">
                <label className="text-[10px] text-gray-500 uppercase tracking-wider mb-2 block">Detected Patterns</label>
                <div className="flex flex-wrap gap-1.5">
                  {currentPrediction.detected_patterns.map((p: any, i: number) => (
                    <span key={i} className={`text-[9px] px-1.5 py-0.5 rounded font-semibold ${p.signal > 0 ? 'bg-green-500/15 text-green-400 border border-green-500/25' : 'bg-red-500/15 text-red-400 border border-red-500/25'}`}>
                      {p.signal > 0 ? '▲' : '▼'} {p.name}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {currentPrediction && !currentPrediction?.detected_patterns?.length && (
              <div className="pt-2 border-t border-[#27272a]">
                <span className="text-[10px] text-gray-600">No strong pattern detected — using technical signals only</span>
              </div>
            )}
          </div>
        </div>

        {/* Accuracy Stats */}
        <div className="glass-panel p-5">
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">Prediction Accuracy</h2>
          <div className="grid grid-cols-3 gap-2 mb-4 text-center">
            <div className="bg-[#18181b] rounded-lg p-2">
              <div className="text-xl font-bold text-white">{resolved.length}</div>
              <div className="text-[10px] text-gray-500 mt-0.5">Total</div>
            </div>
            <div className="bg-green-500/10 rounded-lg p-2">
              <div className="text-xl font-bold text-green-400">{correct}</div>
              <div className="text-[10px] text-gray-500 mt-0.5">Correct</div>
            </div>
            <div className="bg-blue-500/10 rounded-lg p-2">
              <div className="text-xl font-bold text-blue-400">{accuracy}%</div>
              <div className="text-[10px] text-gray-500 mt-0.5">Accuracy</div>
            </div>
          </div>
          {resolved.length > 0 && (
            <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-blue-600 to-blue-400 transition-all duration-500" style={{ width: `${(correct / resolved.length) * 100}%` }} />
            </div>
          )}
        </div>

        {/* Play/Pause */}
        <div className="glass-panel p-4">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`flex items-center justify-center gap-2 w-full py-2.5 rounded-lg transition-colors ${!isPlaying ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' : 'bg-blue-600 text-white hover:bg-blue-500 border border-blue-500'}`}
          >
            {!isPlaying ? <Play size={16} /> : <Pause size={16} />}
            <span className="text-sm">{!isPlaying ? 'Resume Feed' : 'Pause Feed'}</span>
          </button>
        </div>
      </div>

      {/* ── Right: Chart + Log ── */}
      <div className="col-span-9 space-y-4">

        {/* Chart */}
        <div className="glass-panel p-5 h-[420px] flex flex-col">
          <div className="flex justify-between items-center mb-4">
            <div className="flex items-center gap-3">
              <h2 className="text-base font-semibold">Live Price Chart</h2>
              <span className={`font-bold ${activeAsset.color}`}>{activeAsset.icon} {activeSymbol}</span>
              {marketClosed && (
                <span className="text-[10px] bg-orange-500/15 text-orange-400 border border-orange-500/25 px-2 py-0.5 rounded-full animate-pulse">
                  🔒 Market Closed
                </span>
              )}
              <div className="flex bg-[#27272a] rounded-lg p-0.5">
                <button onClick={() => setChartType('candle')} className={`p-1.5 rounded-md transition-colors ${chartType === 'candle' ? 'bg-[#3f3f46] text-white' : 'text-gray-400 hover:text-white'}`}>
                  <BarChart2 size={14} />
                </button>
                <button onClick={() => setChartType('line')} className={`p-1.5 rounded-md transition-colors ${chartType === 'line' ? 'bg-[#3f3f46] text-white' : 'text-gray-400 hover:text-white'}`}>
                  <LineChartIcon size={14} />
                </button>
              </div>
            </div>
            <div className={`text-xl font-mono font-bold ${currentPrice > 0 ? 'text-green-400' : 'text-gray-600'}`}>
              {currentPrice > 0
                ? `$${Number(currentPrice).toLocaleString('en-US', { minimumFractionDigits: 2 })}`
                : '$ -.--'}
            </div>
          </div>

          <div className="flex-1 relative rounded-lg overflow-hidden ring-1 ring-white/5">
            {/* Market Closed Overlay */}
            {marketClosed ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#09090b]/90 backdrop-blur-sm z-10">
                <div className="text-5xl mb-4">🔒</div>
                <div className="text-xl font-bold text-orange-400 mb-2">Market Closed</div>
                <div className="text-sm text-gray-500 mb-1">COMEX Gold Futures</div>
                <div className="text-xs text-gray-600">Trading resumes Sunday 23:00 UTC</div>
                <div className="mt-4 text-[10px] text-gray-700 bg-[#18181b] px-3 py-1.5 rounded-full">
                  Mon–Fri 23:00–22:00 UTC · Daily break 22:00–23:00 UTC
                </div>
              </div>
            ) : (
              <LightweightChart data={liveData} chartType={chartType} />
            )}
          </div>
        </div>

        {/* Prediction History Log + Trading Desk */}
        <div className="grid grid-cols-5 gap-4">
          <div className="col-span-3 glass-panel p-5">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Prediction Log</h2>
            {history.length === 0 ? (
              <div className="text-gray-600 text-sm text-center py-4">Waiting for first prediction cycle...</div>
            ) : (
              <div className="space-y-2 max-h-[300px] overflow-y-auto pr-1">
                {history.map((r) => (
                  <div key={r.id} className={`rounded-lg text-xs overflow-hidden ${r.correct === null ? 'bg-yellow-500/5 border border-yellow-500/20' : r.correct ? 'bg-green-500/10 border border-green-500/20' : 'bg-red-500/10 border border-red-500/20'}`}>
                    <div className="flex items-center justify-between px-3 py-2">
                      <div className="flex items-center gap-2">
                        {r.correct === null ? <Clock size={13} className="text-yellow-400 animate-pulse" /> : r.correct ? <CheckCircle size={13} className="text-green-400" /> : <XCircle size={13} className="text-red-400" />}
                        <span className="text-gray-300">#{r.id}</span>
                        <span className={r.predicted === 'bull' ? 'text-green-400' : 'text-red-400'}>
                          {r.predicted === 'bull' ? '▲ Bull' : '▼ Bear'}
                          <span className="text-gray-500 ml-1">({(r.predicted === 'bull' ? r.bull_prob : r.bear_prob) * 100 | 0}%)</span>
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        {r.actual ? (
                          <span className={r.actual === 'bull' ? 'text-green-400' : 'text-red-400'}>
                            → {r.actual === 'bull' ? '▲ Bull' : '▼ Bear'}
                          </span>
                        ) : (
                          <span className="text-yellow-400 font-mono tabular-nums text-[10px]">
                            {countdown[r.id] !== undefined ? `⏱ ${fmtCountdown(countdown[r.id])}` : '⏳ Resolving'}
                          </span>
                        )}
                        {r.correct !== null && (
                          <span className={`font-semibold ${r.correct ? 'text-green-400' : 'text-red-400'}`}>
                            {r.correct ? '✓ HIT' : '✗ MISS'}
                          </span>
                        )}
                      </div>
                    </div>
                    {r.correct === false && r.reason && (
                      <div className="px-3 pb-2 text-[9px] text-yellow-500/70 italic leading-snug border-t border-red-500/10 pt-1">
                        ✗ {r.reason}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Trading Desk — replaces the 4 metric cards */}
          <div className="col-span-2">
            <TradingPanel
              currentPrice={currentPrice}
              symbol={activeSymbol}
              predictionData={currentPrediction}
              marketClosed={marketClosed}
            />
          </div>
        </div>
      </div>
    </main>
  );
}

