// Reads {weights, cases:[{points, aspect}]} as JSON on stdin; prints features and probabilities.
import { readFileSync } from 'node:fs';
import { landmarkFeatures, parseWeights, predictProba } from '../../backend/static/js/web-asl.js';

const { weights, cases } = JSON.parse(readFileSync(0, 'utf8'));
const model = parseWeights(weights);
const out = cases.map(({ points, aspect }) => ({
  features: Array.from(landmarkFeatures(points, aspect, false)),
  mirrored: Array.from(landmarkFeatures(points, aspect, true)),
  proba: predictProba(model, points, aspect),
}));
process.stdout.write(JSON.stringify(out));
