"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
    DollarSign, TrendingUp, TrendingDown, X, Bot,
    ToggleLeft, ToggleRight, ChevronUp, ChevronDown
} from "lucide-react";

// ─── Types ────────────────────────────────────────────────────────────────────
export type TradeDirection = "buy" | "sell";

export interface ActiveTrade {
    id: number;
    symbol: string;
    direction: TradeDirection;
    units: number;
    entryPrice: number;
    stopLoss: number;        // 0 = none
    exitAfterSecs: number;   // 0 = no auto-exit
    startedAt: number;
    source: "manual" | "auto";
}

export interface ClosedTrade {
    id: number;
    symbol: string;
    direction: TradeDirection;
    units: number;
    entryPrice: number;
    exitPrice: number;
    pnlUsd: number;
    outcome: "sl" | "timer" | "manual";
    closedAt: number;
}

interface TradingPanelProps {
    currentPrice: number;
    symbol: string;
    predictionData: {
        predicted_direction?: "bull" | "bear";
        confidence_score?: number;
        bull_probability?: number;
        bear_probability?: number;
    } | null;
    marketClosed?: boolean;
}

const DEFAULT_UNITS = 10;
let _tradeId = 1;

const fmt = (n: number) => (n >= 0 ? "+" : "") + n.toFixed(2);
const fmtPrice = (n: number) =>
    n >= 100
        ? n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })
        : n.toFixed(4);

const assetTicker = (sym: string) =>
    sym === "BTC-USD" ? "BTC" : sym === "ETH-USD" ? "ETH" : "oz";

function pnlUsd(trade: ActiveTrade, price: number): number {
    const diff = price - trade.entryPrice;
    return trade.direction === "buy" ? diff * trade.units : -diff * trade.units;
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function TradingPanel({
    currentPrice,
    symbol,
    predictionData,
    marketClosed = false,
}: TradingPanelProps) {
    const ticker = assetTicker(symbol);

    const [balance, setBalance] = useState(DEFAULT_UNITS);
    const [activeTrades, setActiveTrades] = useState<ActiveTrade[]>([]);
    const [closedTrades, setClosedTrades] = useState<ClosedTrade[]>([]);
    const [tab, setTab] = useState<"manual" | "auto" | "history">("manual");

    // Manual form
    const [manualDir, setManualDir] = useState<TradeDirection>("buy");
    const [manualUnits, setManualUnits] = useState(1.0);
    const [manualSL, setManualSL] = useState<string>("");
    const [manualExitSecs, setManualExitSecs] = useState(0);

    // Auto form
    const [autoEnabled, setAutoEnabled] = useState(false);
    const [autoConf, setAutoConf] = useState(65);
    const [autoUnits, setAutoUnits] = useState(1.0);
    const [autoExit, setAutoExit] = useState(60);
    const [autoSlPct, setAutoSlPct] = useState(1.0);
    const autoRef = useRef({ enabled: false, conf: 65, units: 1.0, exit: 60, slPct: 1.0, lastFire: 0 });

    useEffect(() => {
        autoRef.current = { enabled: autoEnabled, conf: autoConf, units: autoUnits, exit: autoExit, slPct: autoSlPct, lastFire: autoRef.current.lastFire };
    }, [autoEnabled, autoConf, autoUnits, autoExit, autoSlPct]);

    // Aggregate P&L
    const openPnl = activeTrades.reduce((s, t) => s + pnlUsd(t, currentPrice), 0);
    const totalPnl = closedTrades.reduce((s, t) => s + t.pnlUsd, 0);
    const balanceUsd = currentPrice > 0 ? balance * currentPrice : 0;

    // ── Close trade (safe: no setState inside setState) ──────────────────────
    const closeTrade = useCallback(
        (id: number, exitPrice: number, outcome: ClosedTrade["outcome"]) => {
            setActiveTrades((prev) => {
                const trade = prev.find((t) => t.id === id);
                if (!trade) return prev;
                const pl = pnlUsd(trade, exitPrice);
                const plUnits = exitPrice > 0 ? pl / exitPrice : 0;
                // Schedule balance + history update after this render
                setTimeout(() => {
                    setBalance((b) => Math.max(0, parseFloat((b + trade.units + plUnits).toFixed(6))));
                    setClosedTrades((ct) => [
                        { id: trade.id, symbol: trade.symbol, direction: trade.direction, units: trade.units, entryPrice: trade.entryPrice, exitPrice, pnlUsd: pl, outcome, closedAt: Date.now() },
                        ...ct.slice(0, 29),
                    ]);
                }, 0);
                return prev.filter((t) => t.id !== id);
            });
        },
        []
    );

    // ── Open trade ────────────────────────────────────────────────────────────
    const openTrade = useCallback(
        (dir: TradeDirection, units: number, sl: number, exitSecs: number, src: "manual" | "auto") => {
            if (currentPrice <= 0 || units <= 0) return;
            setBalance((b) => {
                if (b < units) return b;
                const trade: ActiveTrade = {
                    id: _tradeId++,
                    symbol,
                    direction: dir,
                    units,
                    entryPrice: currentPrice,
                    stopLoss: sl,
                    exitAfterSecs: exitSecs,
                    startedAt: Date.now(),
                    source: src,
                };
                // Schedule adding the trade after balance update
                setTimeout(() => setActiveTrades((prev) => [...prev, trade]), 0);
                return parseFloat((b - units).toFixed(6));
            });
        },
        [currentPrice, symbol]
    );

    // ── Monitor active trades ─────────────────────────────────────────────────
    useEffect(() => {
        if (activeTrades.length === 0) return;
        const iv = setInterval(() => {
            const now = Date.now();
            activeTrades.forEach((t) => {
                // Stop-loss check
                if (t.stopLoss > 0) {
                    const hit = t.direction === "buy" ? currentPrice <= t.stopLoss : currentPrice >= t.stopLoss;
                    if (hit) { closeTrade(t.id, t.stopLoss, "sl"); return; }
                }
                // Timer check
                if (t.exitAfterSecs > 0 && (now - t.startedAt) / 1000 >= t.exitAfterSecs) {
                    closeTrade(t.id, currentPrice, "timer"); return;
                }
            });
        }, 1000);
        return () => clearInterval(iv);
    }, [activeTrades, currentPrice, closeTrade]);

    // ── Auto trading ──────────────────────────────────────────────────────────
    useEffect(() => {
        if (!predictionData || !autoRef.current.enabled || marketClosed) return;
        const conf = (predictionData.confidence_score ?? 0) * 100;
        if (conf < autoRef.current.conf) return;
        const now = Date.now();
        if (now - autoRef.current.lastFire < autoRef.current.exit * 1000) return;
        autoRef.current.lastFire = now;
        const dir: TradeDirection = predictionData.predicted_direction === "bear" ? "sell" : "buy";
        const slPct = autoRef.current.slPct / 100;
        const sl = dir === "buy" ? currentPrice * (1 - slPct) : currentPrice * (1 + slPct);
        openTrade(dir, autoRef.current.units, sl, autoRef.current.exit, "auto");
    }, [predictionData, marketClosed, openTrade, currentPrice]);

    // ── Tick for countdowns ────────────────────────────────────────────────────
    const [tick, setTick] = useState(0);
    useEffect(() => {
        const iv = setInterval(() => setTick((t) => t + 1), 1000);
        return () => clearInterval(iv);
    }, []);

    const remSecs = (t: ActiveTrade) =>
        t.exitAfterSecs === 0 ? null : Math.max(0, Math.ceil(t.exitAfterSecs - (Date.now() - t.startedAt) / 1000));

    const fmtTime = (s: number) =>
        s >= 60 ? `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}` : `${s}s`;

    const handleManualTrade = () => {
        if (currentPrice <= 0 || marketClosed || balance < manualUnits) return;
        const sl = manualSL === "" ? 0 : Number(manualSL);
        openTrade(manualDir, manualUnits, sl, manualExitSecs, "manual");
    };

    return (
        <div className="glass-panel p-4 h-full flex flex-col" style={{ minHeight: 0 }}>
            {/* Header */}
            <div className="flex items-center justify-between mb-3 flex-shrink-0">
                <div className="flex items-center gap-2">
                    <DollarSign className="text-yellow-400" size={15} />
                    <span className="font-semibold text-xs tracking-tight">Trading Desk</span>
                    <span className="text-[9px] bg-yellow-500/15 text-yellow-400 border border-yellow-500/25 px-1.5 py-0.5 rounded-full">PAPER</span>
                </div>
                <div className="flex items-center gap-3 text-[10px]">
                    <div className="text-center">
                        <div className="text-gray-500 uppercase" style={{ fontSize: "8px" }}>Balance</div>
                        <div className="font-mono font-bold text-white">{balance.toFixed(3)} {ticker}</div>
                        <div className="text-gray-600" style={{ fontSize: "8px" }}>${balanceUsd.toLocaleString("en-US", { maximumFractionDigits: 0 })}</div>
                    </div>
                    <div className="text-center">
                        <div className="text-gray-500 uppercase" style={{ fontSize: "8px" }}>Open P&L</div>
                        <div className={`font-mono font-bold ${openPnl >= 0 ? "text-green-400" : "text-red-400"}`}>{fmt(openPnl)}$</div>
                    </div>
                    <div className="text-center">
                        <div className="text-gray-500 uppercase" style={{ fontSize: "8px" }}>Total P&L</div>
                        <div className={`font-mono font-bold ${totalPnl >= 0 ? "text-green-400" : "text-red-400"}`}>{fmt(totalPnl)}$</div>
                    </div>
                </div>
            </div>

            {/* Tabs */}
            <div className="flex bg-[#27272a] rounded-lg p-0.5 mb-3 text-[10px] flex-shrink-0">
                {(["manual", "auto", "history"] as const).map((t) => (
                    <button key={t} onClick={() => setTab(t)}
                        className={`flex-1 py-1 rounded-md capitalize transition-colors ${tab === t ? "bg-[#3f3f46] text-white" : "text-gray-400 hover:text-white"}`}>
                        {t === "auto" ? "🤖 Auto" : t === "manual" ? "✋ Manual" : "📜 History"}
                    </button>
                ))}
            </div>

            {/* Tab Body */}
            <div className="flex-1 overflow-y-auto min-h-0">

                {/* Manual */}
                {tab === "manual" && (
                    <div className="space-y-2.5">
                        <div className="grid grid-cols-2 gap-1.5">
                            <button onClick={() => setManualDir("buy")}
                                className={`py-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-1 transition-all ${manualDir === "buy" ? "bg-green-500/20 text-green-300 border border-green-500/40" : "bg-[#27272a] text-gray-400 hover:text-green-400"}`}>
                                <ChevronUp size={13} /> BUY ▲
                            </button>
                            <button onClick={() => setManualDir("sell")}
                                className={`py-2 rounded-lg text-xs font-semibold flex items-center justify-center gap-1 transition-all ${manualDir === "sell" ? "bg-red-500/20 text-red-300 border border-red-500/40" : "bg-[#27272a] text-gray-400 hover:text-red-400"}`}>
                                <ChevronDown size={13} /> SELL ▼
                            </button>
                        </div>
                        <div>
                            <label className="text-[9px] text-gray-500 uppercase tracking-wider block mb-1">Units ({ticker})</label>
                            <input type="number" value={manualUnits}
                                onChange={(e) => setManualUnits(Math.max(0.01, Number(e.target.value)))}
                                className="w-full bg-[#27272a] border border-[#3f3f46] rounded-lg px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-blue-500/50"
                                min={0.01} step={0.1} />
                            <div className="flex gap-1 mt-1">
                                {[0.1, 0.5, 1, 2].map((v) => (
                                    <button key={v} onClick={() => setManualUnits(v)}
                                        className="flex-1 text-[9px] bg-[#27272a] hover:bg-[#3f3f46] rounded py-0.5 text-gray-400">{v}</button>
                                ))}
                            </div>
                        </div>
                        <div>
                            <label className="text-[9px] text-gray-500 uppercase tracking-wider block mb-1">Stop Loss (price)</label>
                            <input type="number" value={manualSL}
                                placeholder={currentPrice > 0 ? fmtPrice(currentPrice * (manualDir === "buy" ? 0.99 : 1.01)) : "—"}
                                onChange={(e) => setManualSL(e.target.value)}
                                className="w-full bg-[#27272a] border border-[#3f3f46] rounded-lg px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-blue-500/50 placeholder-gray-600" />
                        </div>
                        <div>
                            <label className="text-[9px] text-gray-500 uppercase tracking-wider block mb-1">Auto-Exit Time</label>
                            <div className="grid grid-cols-5 gap-1">
                                {([[0, "None"], [60, "1m"], [120, "2m"], [300, "5m"], [600, "10m"]] as [number, string][]).map(([s, l]) => (
                                    <button key={s} onClick={() => setManualExitSecs(s)}
                                        className={`py-1 text-[9px] rounded-md transition-colors ${manualExitSecs === s ? "bg-blue-600 text-white" : "bg-[#27272a] text-gray-400 hover:bg-[#3f3f46]"}`}>
                                        {l}
                                    </button>
                                ))}
                            </div>
                        </div>
                        <div className="flex justify-between text-[10px] px-0.5">
                            <span className="text-gray-500">Entry</span>
                            <span className="font-mono text-white">{currentPrice > 0 ? fmtPrice(currentPrice) : "—"}</span>
                        </div>
                        <button onClick={handleManualTrade}
                            disabled={currentPrice <= 0 || marketClosed || balance < manualUnits}
                            className={`w-full py-2 rounded-lg font-semibold text-xs transition-all ${manualDir === "buy" ? "bg-green-600 hover:bg-green-500" : "bg-red-600 hover:bg-red-500"} text-white disabled:opacity-40 disabled:cursor-not-allowed`}>
                            {marketClosed ? "Market Closed" : `${manualDir === "buy" ? "LONG ▲" : "SHORT ▼"} ${manualUnits} ${ticker}`}
                        </button>
                    </div>
                )}

                {/* Auto */}
                {tab === "auto" && (
                    <div className="space-y-2.5">
                        <div className="flex items-center justify-between p-2.5 rounded-lg bg-[#27272a]">
                            <div>
                                <div className="text-xs font-semibold">{autoEnabled ? "Auto-Trading ON" : "Auto-Trading OFF"}</div>
                                <div className="text-[9px] text-gray-500 mt-0.5">Fires trades based on model</div>
                            </div>
                            <button onClick={() => setAutoEnabled(!autoEnabled)} className="text-blue-400">
                                {autoEnabled ? <ToggleRight size={28} /> : <ToggleLeft size={28} className="text-gray-600" />}
                            </button>
                        </div>
                        <div>
                            <label className="text-[9px] text-gray-500 uppercase tracking-wider block mb-1">Min Confidence — {autoConf}%</label>
                            <input type="range" min={50} max={90} step={1} value={autoConf}
                                onChange={(e) => setAutoConf(Number(e.target.value))} className="w-full accent-blue-500" />
                        </div>
                        <div>
                            <label className="text-[9px] text-gray-500 uppercase tracking-wider block mb-1">Trade Size ({ticker})</label>
                            <input type="number" value={autoUnits}
                                onChange={(e) => setAutoUnits(Math.max(0.01, Number(e.target.value)))}
                                className="w-full bg-[#27272a] border border-[#3f3f46] rounded-lg px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none"
                                min={0.01} step={0.1} />
                        </div>
                        <div>
                            <label className="text-[9px] text-gray-500 uppercase tracking-wider block mb-1">Stop Loss — {autoSlPct.toFixed(1)}%</label>
                            <input type="range" min={0.5} max={5} step={0.5} value={autoSlPct}
                                onChange={(e) => setAutoSlPct(Number(e.target.value))} className="w-full accent-red-500" />
                        </div>
                        <div>
                            <label className="text-[9px] text-gray-500 uppercase tracking-wider block mb-1">Auto-Exit</label>
                            <div className="grid grid-cols-4 gap-1">
                                {([60, 120, 300, 600] as number[]).map((s) => (
                                    <button key={s} onClick={() => setAutoExit(s)}
                                        className={`py-1 text-[9px] rounded-md transition-colors ${autoExit === s ? "bg-blue-600 text-white" : "bg-[#27272a] text-gray-400 hover:bg-[#3f3f46]"}`}>
                                        {s / 60}min
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                {/* History */}
                {tab === "history" && (
                    <div>
                        {closedTrades.length === 0 ? (
                            <div className="text-gray-600 text-xs text-center py-6">No closed trades yet</div>
                        ) : (
                            <div className="space-y-1">
                                {closedTrades.map((t) => (
                                    <div key={t.id} className={`rounded-lg px-2.5 py-1.5 text-[9px] ${t.pnlUsd >= 0 ? "bg-green-500/10 border border-green-500/20" : "bg-red-500/10 border border-red-500/20"}`}>
                                        <div className="flex justify-between">
                                            <span className={t.direction === "buy" ? "text-green-400" : "text-red-400"}>
                                                {t.direction === "buy" ? "▲ LONG" : "▼ SHORT"} {t.units}{ticker}
                                            </span>
                                            <span className={`font-mono font-bold ${t.pnlUsd >= 0 ? "text-green-400" : "text-red-400"}`}>{fmt(t.pnlUsd)}$</span>
                                        </div>
                                        <div className="flex justify-between text-gray-500 mt-0.5">
                                            <span>{fmtPrice(t.entryPrice)} → {fmtPrice(t.exitPrice)}</span>
                                            <span>{t.outcome === "sl" ? "🛑 SL" : t.outcome === "timer" ? "⏱ Timer" : "✋ Manual"}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Open Positions Strip */}
            {activeTrades.length > 0 && (
                <div className="mt-3 border-t border-[#27272a] pt-2 flex-shrink-0">
                    <div className="text-[9px] text-gray-500 uppercase tracking-wider mb-1.5">Open ({activeTrades.length})</div>
                    <div className="space-y-1 max-h-32 overflow-y-auto">
                        {activeTrades.map((t) => {
                            const pl = pnlUsd(t, currentPrice);
                            const rem = remSecs(t);
                            return (
                                <div key={t.id} className={`rounded-lg p-2 border flex items-center gap-2 ${pl >= 0 ? "bg-green-500/8 border-green-500/20" : "bg-red-500/8 border-red-500/20"}`}>
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-1 text-[9px]">
                                            {t.direction === "buy"
                                                ? <TrendingUp size={10} className="text-green-400 flex-shrink-0" />
                                                : <TrendingDown size={10} className="text-red-400 flex-shrink-0" />}
                                            <span className={t.direction === "buy" ? "text-green-300" : "text-red-300"}>
                                                {t.direction === "buy" ? "LONG" : "SHORT"} {t.units}{ticker}
                                            </span>
                                            {t.source === "auto" && <Bot size={8} className="text-blue-400" />}
                                        </div>
                                        <div className="flex gap-2 text-[8px] text-gray-500 mt-0.5">
                                            <span>@{fmtPrice(t.entryPrice)}</span>
                                            {t.stopLoss > 0 && <span className="text-orange-400">SL:{fmtPrice(t.stopLoss)}</span>}
                                            {rem !== null && <span className={rem < 10 ? "text-red-400 animate-pulse" : "text-yellow-400"}>⏱{fmtTime(rem)}</span>}
                                        </div>
                                    </div>
                                    <div className={`text-xs font-mono font-bold flex-shrink-0 ${pl >= 0 ? "text-green-400" : "text-red-400"}`}>
                                        {fmt(pl)}$
                                    </div>
                                    <button onClick={() => closeTrade(t.id, currentPrice, "manual")}
                                        className="text-gray-600 hover:text-red-400 transition-colors flex-shrink-0">
                                        <X size={11} />
                                    </button>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}
