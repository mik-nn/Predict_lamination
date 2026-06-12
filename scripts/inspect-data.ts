import { readFileSync } from "fs";

function inspect() {
  const files = [
    "Data/CGATS/R2_11-4-23.txt",
    "Data/CGATS/R2_11-4-23_lam.txt",
    "Data/CGATS/R2_27-10-23.txt",
    "Data/CGATS/R2_27-10-23_lam.txt",
    "Data/CGATS/R2_13-02-24.txt",
    "Data/CGATS/R2_13-02-24_lam.txt",
    "Data/CGATS/R3_23-4-24.txt",
    "Data/CGATS/R3_23-4-24_lam.txt",
  ];

  for (const f of files) {
    const text = readFileSync(f, "utf-8");
    const lines = text.split(/\r?\n/);
    let dataStart = -1;
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].trim() === "BEGIN_DATA") { dataStart = i; break; }
    }
    const patches: any[] = [];
    for (let i = dataStart + 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line || line === "END_DATA") break;
      const cols = line.split(/\t/);
      patches.push({
        id: cols[0],
        c: parseFloat(cols[1]), m: parseFloat(cols[2]), y: parseFloat(cols[3]), k: parseFloat(cols[4]),
        r380: parseFloat(cols[5])
      });
    }
    const paper = patches.find(p => p.c === 0 && p.m === 0 && p.y === 0 && p.k === 0);
    const hasK = patches.filter(p => p.k > 0);
    const noK = patches.filter(p => p.k === 0);
    const uniqueCMYK = new Set(patches.map(p => p.c+","+p.m+","+p.y+","+p.k));
    console.log(f.split("/").pop() + ": " + patches.length + " patches, paper=" + (paper ? "id="+paper.id+" R380="+paper.r380.toFixed(4) : "NOT FOUND") + ", K>0=" + hasK.length + ", K=0=" + noK.length + ", uniqueCMYK=" + uniqueCMYK.size);
  }
}
inspect();
