import React from 'react';

interface DataTableProps {
  columns: string[];
  rows: Array<Record<string, unknown>>;
}

export const DataTable: React.FC<DataTableProps> = ({ columns, rows }) => {
  if (rows.length === 0) return null;
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx}>
              {columns.map((col) => {
                const val = row[col];
                const display =
                  val === null || val === undefined || val === '' ? '—' : String(val);
                return <td key={col}>{display}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
