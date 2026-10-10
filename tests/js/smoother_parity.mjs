// Reads {frames:[[t, scores|null], ...]} on stdin; prints [[letter|null, confidence], ...].
import { readFileSync } from 'node:fs';
import { LetterSmoother } from '../../backend/static/js/web-asl.js';

const { frames } = JSON.parse(readFileSync(0, 'utf8'));
const smoother = new LetterSmoother();
process.stdout.write(JSON.stringify(frames.map(([t, scores]) => smoother.feed(scores, t))));
