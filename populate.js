// Populate from a = (PI/4) * 1 / 90, tan(a) to a = (PI/4) * 90 / 90, tan(a)
const populate = () => {
  const angles = [];
  const tans = [];
  const frac = 2 ** 12;
  for (let i = 1; i < 91; i++) {
    const a = i * Math.PI / (4 * 90);
    const t = Math.tan(a);
    angles.push(Math.round(frac * a))
    tans.push(Math.round(frac * t))
  }
  console.log("/// Constants used to convert values from cartesian coordinates to polar coordinates");
  console.log("/// Constant angles and tan(a) from a = (PI/4) * 1 / 90, tan(a) to a = (PI/4) * 90 / 90, tan(a)")
  console.log("const ANGLES: [i64; 90] = [")
  angles.forEach(a => console.log(`    ${a}i64,`))
  console.log("];")
  console.log("const TANS: [i64; 90] = [")
  tans.forEach(a => console.log(`    ${a}i64,`))
  console.log("];")
  const pi = Math.round(frac * Math.PI);
  const pi2 = Math.round(frac * (Math.PI * Math.PI));
  const pix2 = Math.round(2 * frac * Math.PI);
  console.log("/// PI << 12, (PI ** 2) << 12, (PI * 2) << 12");
  console.log(`const PIS12: i64 = ${pi}i64;`);
  console.log(`const PIS12_SQUARE: i64 = ${pi2}i64;`);
  console.log(`const PIS12_DOUBLE: i64 = ${pix2}i64;`);
}

populate();
