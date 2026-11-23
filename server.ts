import path from 'node:path';
import express from 'express';
import { createServer as createViteServer } from 'vite';
import { createApiApp } from './server/app.js';
import { getDataStore } from './server/dataLoader.js';
import { structuralAnalogueEvidence } from './server/chemistry.js';

const PORT = Number(process.env.PORT || 3000);
const HOST = '0.0.0.0';

async function startServer() {
  const app = createApiApp();

  const isProd = process.env.NODE_ENV === 'production';
  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.join(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, HOST, () => {
    console.log(`MEPRank Research Dashboard server listening on http://${HOST}:${PORT}`);
    setTimeout(() => {
      try {
        getDataStore();
        structuralAnalogueEvidence('CID000444795', 'CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CC(=O)O)C)C');
        console.log('MEPRank datasets and analogue fingerprint index warmed up.');
      } catch (err) {
        console.warn('Warmup warning:', err);
      }
    }, 50);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
