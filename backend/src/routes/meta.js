import { Router } from 'express';
import { readFileSync } from 'fs';
import path from 'path';


const router = Router();
const dataDir = path.resolve(process.cwd(), 'data');


router.get('/subdivisions', (_req, res) => {
const p = path.join(dataDir, 'subdivisions.json');
const subs = JSON.parse(readFileSync(p, 'utf-8'));
res.json(subs);
});


export default router;