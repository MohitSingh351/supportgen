// Small Recharts wrappers that follow the dataviz rules: thin marks, rounded data-ends,
// recessive grid, one axis, a hover tooltip on every chart, text in ink colours.
import {
  Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

const axisProps = {
  stroke: "var(--axis)", tick: { fill: "var(--muted)", fontSize: 12 }, tickLine: false,
};

function TooltipBox({ active, payload, label, format = (v) => v }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tt">
      <b>{label}</b>
      {payload.map((p) => (
        <div key={p.dataKey}>
          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: p.color, marginRight: 6 }} />
          {p.name}: <span className="num">{format(p.value)}</span>
        </div>
      ))}
    </div>
  );
}

/** Horizontal bars for a {label: count} distribution (single series → no legend). */
export function DistributionBars({ data, color = "var(--seq)", height }) {
  const rows = Object.entries(data || {}).map(([name, value]) => ({ name: name.charAt(0).toUpperCase() + name.slice(1), value }));
  if (!rows.length) return <p className="muted small">No data yet.</p>;
  const h = height || Math.max(120, rows.length * 34 + 30);
  return (
    <ResponsiveContainer width="100%" height={h}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }} barCategoryGap={6}>
        <CartesianGrid horizontal={false} stroke="var(--grid)" />
        <XAxis type="number" allowDecimals={false} {...axisProps} />
        <YAxis type="category" dataKey="name" width={170} {...axisProps} axisLine={false} />
        <Tooltip cursor={{ fill: "var(--plane)" }} content={<TooltipBox />} />
        <Bar dataKey="value" name="Tickets" fill={color} radius={[0, 4, 4, 0]} maxBarSize={22} isAnimationActive={false}
             label={{ position: "right", fill: "var(--ink-2)", fontSize: 12 }} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Grouped vertical bars comparing named series across metrics. */
export function GroupedBars({ rows, series, height = 280, format = (v) => v.toFixed(3) }) {
  return (
    <>
      <div className="legend">
        {series.map((s) => <span key={s.key}><i style={{ background: s.color }} />{s.name}</span>)}
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 4, left: 0 }} barGap={2} barCategoryGap="22%">
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis dataKey="metric" {...axisProps} interval={0} tick={{ fill: "var(--muted)", fontSize: 11 }} />
          <YAxis {...axisProps} axisLine={false} domain={[0, "auto"]} />
          <Tooltip cursor={{ fill: "var(--plane)" }} content={<TooltipBox format={format} />} />
          {series.map((s) => (
            <Bar key={s.key} dataKey={s.key} name={s.name} fill={s.color} radius={[4, 4, 0, 0]} maxBarSize={36} isAnimationActive={false} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </>
  );
}

/** Line chart for loss curves / time series. */
export function Lines({ rows, xKey, series, height = 260, xLabel, xType = "number", xTicks, format = (v) => v.toFixed(3) }) {
  return (
    <>
      {series.length > 1 && (
        <div className="legend">
          {series.map((s) => <span key={s.key}><i style={{ background: s.color }} />{s.name}</span>)}
        </div>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={rows} margin={{ top: 8, right: 16, bottom: xLabel ? 18 : 4, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--grid)" />
          <XAxis dataKey={xKey} type={xType} domain={xType === "number" ? ["dataMin", "dataMax"] : undefined} ticks={xTicks} {...axisProps}
                 label={xLabel ? { value: xLabel, position: "insideBottom", offset: -10, fill: "var(--muted)", fontSize: 12 } : undefined} />
          <YAxis {...axisProps} axisLine={false} domain={["auto", "auto"]} tickFormatter={(v) => +v.toFixed(2)} />
          <Tooltip content={<TooltipBox format={format} />} labelFormatter={(v) => (xLabel ? `${xLabel} ${+Number(v).toFixed(2)}` : v)} />
          {series.map((s) => (
            <Line key={s.key} data={s.data} dataKey={s.key} name={s.name} stroke={s.color} strokeWidth={2}
                  dot={s.dots ? { r: 5, strokeWidth: 2, stroke: "var(--paper)", fill: s.color } : false}
                  activeDot={{ r: 5 }} connectNulls isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </>
  );
}
