import React from 'react';

interface BarChartItem {
  label: string;
  value: number;
}

export const BarChart: React.FC<{ items: BarChartItem[] }> = ({ items }) => {
  if (items.length === 0) return null;
  const max = Math.max(...items.map((i) => i.value), 1);

  return (
    <div className="chart-box">
      {items.map((item) => {
        const pct = Math.max(0, Math.min(100, (item.value / max) * 100));
        return (
          <div className="bar-row" key={item.label}>
            <span title={item.label}>{item.label}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${pct}%` }} />
            </div>
            <span className="bar-val">{item.value.toLocaleString()}</span>
          </div>
        );
      })}
    </div>
  );
};
