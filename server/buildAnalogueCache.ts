import fs from 'node:fs';
import path from 'node:path';
import { computeAnaloguePool } from './chemistry.js';

const start = Date.now();
const pool = computeAnaloguePool();
const outPath = path.join(process.cwd(), 'data', 'analogue_fingerprints.json');
fs.writeFileSync(outPath, JSON.stringify(pool));
const bytes = fs.statSync(outPath).size;
console.log(
  `Wrote ${pool.length} analogue fingerprints to ${outPath} (${(bytes / 1024).toFixed(1)} KB) in ${Date.now() - start} ms.`
);
