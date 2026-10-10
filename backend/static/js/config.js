/** Constants: colours, limits, the ASL alphabet. No logic. */

export const MAX_WIDTH = 640;               // frames are downscaled to this width before upload
export const JPEG_QUALITY = 0.7;
export const REQUEST_TIMEOUT_MS = 60000;
export const MAX_FACES = 4;
export const MAX_TRACKS = 60;
export const SPARK_POINTS = 40;
export const DEFAULT_MIN_CONF = 0.35;

export const EXPRESSIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised'];

/** Fixed colour per object class (a hash of the class name picks the slot). Brand blue and cyan come first. */
export const PALETTE = ['#2563eb', '#06b6d4', '#f59e0b', '#10b981', '#ec4899', '#8b5cf6',
  '#f97316', '#14b8a6', '#ef4444', '#84cc16', '#6366f1', '#eab308'];
export const EMOTION_COLORS = {
  angry: '#ef4444', disgust: '#84cc16', fear: '#8b5cf6', happy: '#f59e0b',
  neutral: '#64748b', sad: '#3b82f6', surprised: '#f97316',
};
/** Hands are drawn over photos, so they use a fixed bright colour in both themes. */
export const SIGN_COLOR = '#06b6d4';
export const HAND_LINKS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

export const ASL_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
/** Letters that need motion; the classifier reads one still frame, so they are marked and never picked for "Try". */
export const MOTION_LETTERS = ['J', 'Z'];
export const ASL_IMG = (letter) => `/static/asl/${letter}.svg`;
