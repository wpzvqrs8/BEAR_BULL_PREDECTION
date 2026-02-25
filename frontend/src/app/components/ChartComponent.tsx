"use client";

import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, CrosshairMode, ISeriesApi, IChartApi } from 'lightweight-charts';

interface ChartComponentProps {
    data: any;
    chartType: 'candle' | 'line';
}

export default function ChartComponent({ data, chartType }: ChartComponentProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<any> | null>(null);
    // Cache all candle data so chart-type switch re-hydrates without a network call
    const allDataRef = useRef<any[]>([]);

    // ── Initialize chart once on mount ────────────────────────────────────
    useEffect(() => {
        if (!containerRef.current) return;

        const chart = createChart(containerRef.current, {
            width: containerRef.current.clientWidth,
            height: containerRef.current.clientHeight,
            layout: {
                background: { type: ColorType.Solid, color: '#09090b' },
                textColor: '#a1a1aa',
            },
            grid: {
                vertLines: { color: '#27272a' },
                horzLines: { color: '#27272a' },
            },
            crosshair: { mode: CrosshairMode.Normal },
            rightPriceScale: { borderColor: '#3f3f46' },
            timeScale: {
                borderColor: '#3f3f46',
                timeVisible: true,
                secondsVisible: false,
            },
        });
        chartRef.current = chart;

        seriesRef.current = chart.addCandlestickSeries({
            upColor: '#10b981',
            downColor: '#ef4444',
            borderUpColor: '#10b981',
            borderDownColor: '#ef4444',
            wickUpColor: '#10b981',
            wickDownColor: '#ef4444',
        });

        const ro = new ResizeObserver(() => {
            if (containerRef.current) {
                chart.applyOptions({
                    width: containerRef.current.clientWidth,
                    height: containerRef.current.clientHeight,
                });
            }
        });
        ro.observe(containerRef.current);

        return () => {
            ro.disconnect();
            chart.remove();
            chartRef.current = null;
            seriesRef.current = null;
        };
    }, []);

    // ── Switch between candlestick / line — preserve existing data ────────
    useEffect(() => {
        if (!chartRef.current) return;

        try { if (seriesRef.current) chartRef.current.removeSeries(seriesRef.current); } catch (_) { }

        if (chartType === 'candle') {
            seriesRef.current = chartRef.current.addCandlestickSeries({
                upColor: '#10b981', downColor: '#ef4444',
                borderUpColor: '#10b981', borderDownColor: '#ef4444',
                wickUpColor: '#10b981', wickDownColor: '#ef4444',
            });
        } else {
            seriesRef.current = chartRef.current.addLineSeries({
                color: '#3b82f6',
                lineWidth: 2,
            });
        }

        // Re-apply cached data so the toggle doesn't blank the chart
        if (allDataRef.current.length > 0) {
            const rows = allDataRef.current.map((d: any) =>
                chartType === 'candle'
                    ? { time: d.time, open: d.open, high: d.high, low: d.low, close: d.close }
                    : { time: d.time, value: d.close }
            );
            try { seriesRef.current!.setData(rows); } catch (_) { }
        }
    }, [chartType]);

    // ── Feed data into the series whenever `data` prop changes ────────────
    useEffect(() => {
        if (!data || !seriesRef.current) return;

        if (data.type === 'history' && Array.isArray(data.data)) {
            // Full history load — replace everything
            const sorted = [...data.data].sort((a: any, b: any) => a.time - b.time);
            allDataRef.current = sorted;

            const rows = sorted.map((d: any) =>
                chartType === 'candle'
                    ? { time: d.time, open: d.open, high: d.high, low: d.low, close: d.close }
                    : { time: d.time, value: d.close }
            );
            try {
                seriesRef.current!.setData(rows);
                chartRef.current?.timeScale().fitContent();
            } catch (e) {
                console.warn('setData error', e);
            }

        } else if (data.type === 'candle' || data.type === 'chart_candle') {
            // Single bar update:
            //  • chart_candle forming:true  → updates the current rightmost bar in-place (live OHLC)
            //  • chart_candle forming:false → appends a completed minute bar
            //  • candle                     → appends the resolved prediction bar
            const point =
                chartType === 'candle'
                    ? { time: data.time, open: data.open, high: data.high, low: data.low, close: data.close }
                    : { time: data.time, value: data.close };

            // Keep allDataRef in sync
            const idx = allDataRef.current.findIndex((d: any) => d.time === data.time);
            if (idx >= 0) {
                allDataRef.current[idx] = { ...allDataRef.current[idx], ...data };
            } else {
                allDataRef.current.push({ ...data });
            }

            try {
                // series.update() merges when timestamp === last bar; appends otherwise
                seriesRef.current!.update(point as any);
                if (data.forming) {
                    chartRef.current?.timeScale().scrollToRealTime();
                }
            } catch (e) {
                console.warn('update error', e);
            }
        }
    }, [data, chartType]);

    return <div ref={containerRef} className="absolute inset-0" />;
}
