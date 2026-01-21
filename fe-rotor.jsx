// Neural pulses with AI rotor classification
const rotorX = width * 0.5;
const rotorY = height * 0.65;
const rotorRadius = 22;

// Draw AI rotor
ctx.beginPath();
ctx.arc(rotorX, rotorY, rotorRadius, 0, Math.PI * 2);
ctx.strokeStyle = "rgba(255,0,0,0.4)";
ctx.lineWidth = 2;
ctx.stroke();

// Spinning blades
for (let b = 0; b < 3; b++) {
  const angle = frame * 0.08 + b * (Math.PI * 2 / 3);
  ctx.beginPath();
  ctx.moveTo(rotorX, rotorY);
  ctx.lineTo(
    rotorX + Math.cos(angle) * rotorRadius,
    rotorY + Math.sin(angle) * rotorRadius
  );
  ctx.strokeStyle = "rgba(255,0,0,0.5)";
  ctx.stroke();
}

// Pulses
for (let i = 0; i < 40; i++) {
  const x = (i * 40 + frame * 1.5) % width;
  const y = height * 0.65 - Math.sin((x + frame) / 40) * 40;

  let color = "black";

  // After rotor → classify into G/A/R
  if (x > rotorX + rotorRadius) {
    const phase = i % 3;
    if (phase === 0) color = "#2ecc71";      // Green
    if (phase === 1) color = "#f39c12";      // Amber
    if (phase === 2) color = "#e74c3c";      // Red
  }

  ctx.beginPath();
  ctx.arc(x, y, 2.4, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}
/////////////////////////////////////
const engineX = width * 0.5;
const engineY = height * 0.65;
const engineRadius = 26;

// Outer ring
ctx.beginPath();
ctx.arc(engineX, engineY, engineRadius, 0, Math.PI * 2);
ctx.strokeStyle = "rgba(255,0,0,0.5)";
ctx.lineWidth = 2;
ctx.stroke();

// Inner ring
ctx.beginPath();
ctx.arc(engineX, engineY, engineRadius - 8, 0, Math.PI * 2);
ctx.strokeStyle = "rgba(255,0,0,0.3)";
ctx.lineWidth = 2;
ctx.stroke();

// Spinning spokes (engine blades)
for (let b = 0; b < 6; b++) {
  const angle = frame * 0.06 + b * (Math.PI * 2 / 6);
  ctx.beginPath();
  ctx.moveTo(engineX, engineY);
  ctx.lineTo(
    engineX + Math.cos(angle) * engineRadius,
    engineY + Math.sin(angle) * engineRadius
  );
  ctx.strokeStyle = "rgba(255,0,0,0.6)";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

// Core glow
ctx.beginPath();
ctx.arc(engineX, engineY, 6, 0, Math.PI * 2);
ctx.fillStyle = "rgba(255,0,0,0.7)";
ctx.fill();

for (let i = 0; i < 40; i++) {
  const x = (i * 40 + frame * 1.5) % width;
  const y = height * 0.65 - Math.sin((x + frame) / 40) * 40;

  let color = "black";

  // After engine → classify
  if (x > engineX + engineRadius) {
    const phase = i % 3;
    if (phase === 0) color = "#2ecc71";      // Green
    if (phase === 1) color = "#f39c12";      // Amber
    if (phase === 2) color = "#e74c3c";      // Red
  }

  ctx.beginPath();
  ctx.arc(x, y, 2.4, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

///////////////////////////////////
const gearX = width * 0.5;
const gearY = height * 0.65;
const gearRadius = 26;
const teeth = 10;
const toothDepth = 6;
const rotation = frame * 0.05;

// Draw gear teeth
for (let t = 0; t < teeth; t++) {
  const angle = rotation + t * (Math.PI * 2 / teeth);
  const innerR = gearRadius;
  const outerR = gearRadius + toothDepth;

  const x1 = gearX + Math.cos(angle) * innerR;
  const y1 = gearY + Math.sin(angle) * innerR;
  const x2 = gearX + Math.cos(angle) * outerR;
  const y2 = gearY + Math.sin(angle) * outerR;

  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = "rgba(0,0,0,0.7)";
  ctx.lineWidth = 2;
  ctx.stroke();
}

// Outer gear ring
ctx.beginPath();
ctx.arc(gearX, gearY, gearRadius, 0, Math.PI * 2);
ctx.strokeStyle = "rgba(0,0,0,0.6)";
ctx.lineWidth = 2;
ctx.stroke();

// Inner hub
ctx.beginPath();
ctx.arc(gearX, gearY, gearRadius - 10, 0, Math.PI * 2);
ctx.strokeStyle = "rgba(0,0,0,0.4)";
ctx.lineWidth = 1.5;
ctx.stroke();

// Center axle
ctx.beginPath();
ctx.arc(gearX, gearY, 4, 0, Math.PI * 2);
ctx.fillStyle = "rgba(0,0,0,0.8)";
ctx.fill();

// Optional glow (subtle)
ctx.beginPath();
ctx.arc(gearX, gearY, gearRadius + 2, 0, Math.PI * 2);
ctx.strokeStyle = "rgba(255,0,0,0.15)";
ctx.lineWidth = 1;
ctx.stroke();
