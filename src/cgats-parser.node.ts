// Node.js only: reads CGATS from file path
import { readFileSync } from "node:fs";
import { Patch } from "./types.js";
import { parseCgatsText } from "./cgats-parser.ts";

export { parseCgatsText };

export function parseCgatsFile(filePath: string): Patch[] {
  const text = readFileSync(filePath, "utf-8");
  return parseCgatsText(text);
}
